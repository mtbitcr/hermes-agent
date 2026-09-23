"""``kanban_review_findings`` — real registry dispatch, real board, real git.

Unlike ``tests/hermes_cli/test_kanban_review_findings.py`` (which drives the
kernel entry point ``kanban_db.submit_review_findings`` directly), these
tests dispatch through the REAL tool registry — ``registry.get_entry(name)``
— and call the registered handler exactly as a worker's tool call would, so
a missing registration or a missing toolset entry fails the test. Every test
runs under the exact worker-dispatch environment
(``HERMES_KANBAN_DB``/``HERMES_KANBAN_BOARD``/``HERMES_KANBAN_TASK``/
``HERMES_HOME``) the dispatcher injects, on a real disposable board DB and a
real git worktree so ``_latest_review_head_provenance`` returns a real commit.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

REVIEWER = "reviewer-worker"
IMPLEMENTER = "impl-worker"


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        [
            "git", "-C", str(cwd),
            "-c", "user.name=Test User",
            "-c", "user.email=test@example.com",
            "-c", "commit.gpgsign=false",
            *args,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _finding(**overrides) -> dict:
    base = {
        "severity": "major",
        "file": "src/app.py",
        "lines": "10-20",
        "problem": "off-by-one in the loop bound",
        "impact": "drops the last item in the batch",
        "smallest_fix": "use `<=` instead of `<` in the range check",
    }
    base.update(overrides)
    return base


@pytest.fixture
def repo(tmp_path) -> Path:
    origin = tmp_path / "repo"
    origin.mkdir()
    subprocess.run(
        ["git", "init", "-b", "main", str(origin)],
        check=True, capture_output=True, text=True,
    )
    (origin / "README.md").write_text("base\n", encoding="utf-8")
    _git(origin, "add", "README.md")
    _git(origin, "commit", "-m", "init")
    return origin


@pytest.fixture
def board(tmp_path, monkeypatch, repo):
    """Real disposable board with a task claimed from the review lane.

    Builds the card through the real ``kanban_db`` API (create -> implementer
    claims -> commits into a real git worktree -> request_review -> reviewer
    claims from the review lane), then pins the exact dispatcher environment
    the reviewer's own tool call would run under.
    """
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("HERMES_SESSION_ID", raising=False)

    from hermes_cli import kanban_db as kb
    import tools.kanban_tools  # noqa: F401 - ensure registered

    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    conn = kb.connect()
    tid = kb.create_task(
        conn,
        title="reviewed task",
        assignee=IMPLEMENTER,
        owned_paths=["src"],
        workspace_kind="worktree",
        workspace_path=str(repo),
        branch_name="feature/review-findings",
    )
    run = kb.claim_task(conn, tid, claimer=f"{IMPLEMENTER}:1")
    assert run is not None
    workspace, branch = kb._resolve_worktree_workspace(run)
    kb.set_workspace_path(conn, tid, workspace)
    kb.set_branch_name(conn, tid, branch)
    kb.record_worktree_base(conn, tid, workspace)
    (workspace / "src").mkdir(parents=True, exist_ok=True)
    (workspace / "src" / "app.py").write_text("ok = True\n", encoding="utf-8")
    _git(workspace, "add", "src/app.py")
    _git(workspace, "commit", "-m", "feat: implement")
    head = _git(workspace, "rev-parse", "HEAD")

    assert kb.request_review(
        conn, tid, summary="ready", reviewer=REVIEWER,
        expected_run_id=run.current_run_id,
    )
    review = kb.claim_review_task(conn, tid, claimer=f"{REVIEWER}:1")
    assert review is not None

    monkeypatch.setenv("HERMES_KANBAN_DB", str(kb.kanban_db_path()))
    monkeypatch.setenv("HERMES_KANBAN_BOARD", kb.DEFAULT_BOARD)
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(review.current_run_id))

    try:
        yield kb, conn, tid, head, review
    finally:
        conn.close()


def _dispatch(args: dict) -> dict:
    """Look ``kanban_review_findings`` up in the real registry and call it."""
    from tools.registry import registry

    entry = registry.get_entry("kanban_review_findings")
    assert entry is not None, "kanban_review_findings is not registered"
    assert entry.toolset == "kanban"
    return json.loads(entry.handler(args))


# ---------------------------------------------------------------------------
# 1. Empty array -> pass
# ---------------------------------------------------------------------------


def test_empty_findings_array_approves_the_task(board):
    kb, conn, tid, head, review = board
    out = _dispatch({"findings": []})
    assert out.get("ok") is True, out
    assert out["outcome"] == "passed"

    task = kb.get_task(conn, tid)
    assert task is not None
    assert task.status == "done"


# ---------------------------------------------------------------------------
# 2. Ordinary finding -> handed back to the original implementer
# ---------------------------------------------------------------------------


def test_one_finding_hands_back_to_the_original_implementer(board):
    kb, conn, tid, head, review = board
    out = _dispatch({"findings": [_finding(candidate_digest=head)]})
    assert out.get("ok") is True, out
    assert out["outcome"] == "handed_back"
    assert out["implementer"] == IMPLEMENTER
    assert out.get("attachment_id")

    task = kb.get_task(conn, tid)
    assert task is not None
    assert task.status == "ready"
    assert task.assignee == IMPLEMENTER

    attachments = kb.list_attachments(conn, tid)
    assert len(attachments) == 1
    assert attachments[0].id == out["attachment_id"]


# ---------------------------------------------------------------------------
# 3. Same findings replayed against the unchanged candidate -> owner-blocked
# ---------------------------------------------------------------------------


def test_repeated_identical_findings_block_for_owner_decision(board, monkeypatch):
    kb, conn, tid, head, review = board
    first = _dispatch({"findings": [_finding(candidate_digest=head)]})
    assert first["outcome"] == "handed_back"

    # The implementer re-claims without changing anything, then the SAME
    # reviewer looks again at the SAME candidate.
    implementer_run = kb.claim_task(conn, tid, claimer=f"{IMPLEMENTER}:2")
    assert implementer_run is not None
    assert kb.request_review(
        conn, tid, summary="unchanged", reviewer=REVIEWER,
        expected_run_id=implementer_run.current_run_id,
    )
    review2 = kb.claim_review_task(conn, tid, claimer=f"{REVIEWER}:2")
    assert review2 is not None
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(review2.current_run_id))

    second = _dispatch({"findings": [_finding(candidate_digest=head)]})
    assert second.get("ok") is True, second
    assert second["outcome"] == "owner_decision_blocked"

    task = kb.get_task(conn, tid)
    assert task is not None
    assert task.status == "blocked"
    # The implementer was not re-run: still parked on the reviewer's block,
    # not routed back to a fresh implementer run.
    assert task.assignee == REVIEWER


# ---------------------------------------------------------------------------
# 4. Malformed / missing findings -> refusal, no write
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("args_builder", [
    lambda: {},
    lambda: {"findings": None},
    lambda: {"findings": {"severity": "major"}},
    lambda: {"findings": "not a list"},
])
def test_malformed_or_missing_findings_is_refused_with_no_write(board, args_builder):
    kb, conn, tid, head, review = board
    before = kb.get_task(conn, tid)
    before_attachments = kb.list_attachments(conn, tid)

    out = _dispatch(args_builder())
    assert out.get("ok") is not True
    assert out.get("error")

    after = kb.get_task(conn, tid)
    assert after.status == before.status
    assert after.assignee == before.assignee
    assert kb.list_attachments(conn, tid) == before_attachments


# ---------------------------------------------------------------------------
# 5. candidate_digest disagreeing with the kernel-parked head -> refusal
# ---------------------------------------------------------------------------


def test_mismatched_candidate_digest_is_refused_with_no_write(board):
    kb, conn, tid, head, review = board
    before = kb.get_task(conn, tid)

    out = _dispatch({
        "findings": [_finding(candidate_digest="0" * 40)],
        "candidate_digest": "0" * 40,
    })
    assert out.get("ok") is not True
    assert out.get("error")
    assert "candidate_digest" in out["error"]

    after = kb.get_task(conn, tid)
    assert after.status == before.status
    assert after.assignee == before.assignee
    assert kb.list_attachments(conn, tid) == []


# ---------------------------------------------------------------------------
# 6. Ended/old run -> refusal, no write
# ---------------------------------------------------------------------------


def test_ended_or_stale_run_is_refused_with_no_write(board, monkeypatch):
    kb, conn, tid, head, review = board
    before = kb.get_task(conn, tid)

    # A run id the dispatcher no longer recognizes as current.
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(review.current_run_id + 999))
    out = _dispatch({"findings": [_finding(candidate_digest=head)]})
    assert out.get("ok") is not True
    assert out.get("error")

    after = kb.get_task(conn, tid)
    assert after.status == before.status
    assert after.assignee == before.assignee
    assert kb.list_attachments(conn, tid) == []


def test_task_not_running_is_refused_with_no_write(board, monkeypatch):
    kb, conn, tid, head, review = board
    # Approve the task first, so its run is no longer active.
    approved = _dispatch({"findings": []})
    assert approved["outcome"] == "passed"

    out = _dispatch({"findings": [_finding(candidate_digest=head)]})
    assert out.get("ok") is not True
    assert out.get("error")

    after = kb.get_task(conn, tid)
    assert after.status == "done"


# ---------------------------------------------------------------------------
# 7. Another task id / another board -> refusal (own-task/own-board boundary)
# ---------------------------------------------------------------------------


def test_another_task_id_is_refused(board):
    kb, conn, tid, head, review = board
    out = _dispatch({"task_id": "t_notmine00", "findings": []})
    assert out.get("ok") is not True
    assert out.get("error")
    assert "scoped to task" in out["error"]

    task = kb.get_task(conn, tid)
    assert task is not None
    assert task.status == "running"


def test_another_board_is_refused(board, tmp_path, monkeypatch):
    kb, conn, tid, head, review = board

    other_board = "other-board"
    kb.create_board(other_board)
    # kanban_db_path() honours HERMES_KANBAN_DB before its own board arg, so
    # the other board's real path has to be resolved via board_dir() while
    # the current pin is still in effect, not through kanban_db_path().
    other_db_path = kb.board_dir(other_board) / "kanban.db"
    assert other_db_path != Path(os.environ["HERMES_KANBAN_DB"])

    monkeypatch.setenv("HERMES_KANBAN_DB", str(other_db_path))
    monkeypatch.setenv("HERMES_KANBAN_BOARD", other_board)

    out = _dispatch({"findings": [_finding(candidate_digest=head)]})
    assert out.get("ok") is not True
    assert out.get("error")

    own_task = kb.get_task(conn, tid)
    assert own_task is not None
    assert own_task.status == "running"


def test_mismatched_board_argument_is_refused_without_repointing_pins(board):
    """A caller-supplied ``board`` argument that disagrees with the board the
    worker is actually pinned to must be refused, even though the trusted
    env pins (``HERMES_KANBAN_DB``/``HERMES_KANBAN_BOARD``/``HERMES_KANBAN_TASK``/
    ``HERMES_KANBAN_RUN_ID``) are left completely untouched here.

    ``kb.connect`` honours ``HERMES_KANBAN_DB`` before its own ``board``
    argument, so a naive implementation silently reinterprets a mismatched
    ``board`` as the worker's own pinned board and approves the worker's OWN
    task instead of refusing the mismatch. Unlike
    ``test_another_board_is_refused`` above (which repoints the pins
    themselves), this test proves the tool call's own ``board`` argument is
    validated against the connection actually opened.
    """
    kb, conn, tid, head, review = board
    kb.create_board("different-review-board")

    from tools.registry import registry

    out = json.loads(registry.dispatch("kanban_review_findings", {
        "task_id": tid,
        "findings": [],
        "candidate_digest": head,
        "board": "different-review-board",
    }))
    assert out.get("ok") is not True
    assert out.get("error")

    task = kb.get_task(conn, tid)
    assert task is not None
    assert task.status == "running"
    assert kb.list_attachments(conn, tid) == []


def test_matching_own_board_argument_still_approves(board):
    """The benign counterpart: passing ``board`` equal to the worker's own
    pinned board slug is not a mismatch and must keep approving exactly as
    an omitted ``board`` does — the fix refuses only a genuine mismatch.
    """
    kb, conn, tid, head, review = board

    out = _dispatch({
        "task_id": tid,
        "findings": [],
        "candidate_digest": head,
        "board": kb.DEFAULT_BOARD,
    })
    assert out.get("ok") is True, out
    assert out["outcome"] == "passed"

    task = kb.get_task(conn, tid)
    assert task is not None
    assert task.status == "done"


def test_mismatched_board_argument_is_refused_before_any_connection(
    board, monkeypatch
):
    """A mismatched explicit ``board`` must be refused before ``_connect``
    is ever reached.

    Every connection ``_connect`` opens is write-capable — it enables WAL
    and, on first open, runs schema creation / additive migrations, and
    even the already-initialized fast path still executes WAL and
    persistence-related pragmas. A rejected explicit target must never
    open a board at all, so this replaces ``_connect`` with a sentinel
    that fails the test if it is ever reached for a foreign board.
    """
    kb, conn, tid, head, review = board
    kb.create_board("different-review-board")

    import tools.kanban_tools as kt
    from tools.registry import registry

    called = []

    def _sentinel_connect(*args, **kwargs):
        called.append((args, kwargs))
        raise AssertionError(
            "write-capable connect reached before refusal"
        )

    monkeypatch.setattr(kt, "_connect", _sentinel_connect)

    out = json.loads(registry.dispatch("kanban_review_findings", {
        "task_id": tid,
        "findings": [],
        "candidate_digest": head,
        "board": "different-review-board",
    }))

    assert called == []
    assert out.get("ok") is not True
    assert out.get("error")

    task = kb.get_task(conn, tid)
    assert task is not None
    assert task.status == "running"


def test_matching_board_argument_still_opens_the_connection(
    board, monkeypatch
):
    """The benign control: the worker's own pinned board slug is not a
    mismatch and must still reach ``_connect`` and approve normally — the
    new pre-connect gate refuses only a genuine mismatch.
    """
    kb, conn, tid, head, review = board

    import tools.kanban_tools as kt
    from tools.registry import registry

    real_connect = kt._connect
    called = []

    def _recording_connect(*args, **kwargs):
        called.append((args, kwargs))
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(kt, "_connect", _recording_connect)

    out = json.loads(registry.dispatch("kanban_review_findings", {
        "task_id": tid,
        "findings": [],
        "candidate_digest": head,
        "board": kb.DEFAULT_BOARD,
    }))

    assert called != []
    assert out.get("ok") is True, out
    assert out["outcome"] == "passed"

    task = kb.get_task(conn, tid)
    assert task is not None
    assert task.status == "done"


@pytest.mark.parametrize("pin", [None, "", "   ", "Not A Slug!", "-leading"])
@pytest.mark.parametrize("selector", [None, "different-review-board"])
def test_unusable_board_pin_is_refused_before_any_connection(
    board, monkeypatch, pin, selector
):
    """An unusable HERMES_KANBAN_BOARD (absent, empty, whitespace, or malformed)
    must trigger a pre-connection refusal, regardless of the explicit board selector.
    Without this guard the pre-connection check is skipped for every bad pin value
    and write-capable _connect is reached; every connection runs schema/WAL writes,
    so opening one at all is already a side-effect that must be prevented.
    """
    kb, conn, tid, head, review = board
    kb.create_board("different-review-board")
    import tools.kanban_tools as kt
    from tools.registry import registry

    if pin is None:
        monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    else:
        monkeypatch.setenv("HERMES_KANBAN_BOARD", pin)

    called = []

    def _sentinel_connect(*args, **kwargs):
        called.append((args, kwargs))
        raise AssertionError("write-capable connect reached before refusal")

    monkeypatch.setattr(kt, "_connect", _sentinel_connect)

    args = {"task_id": tid, "findings": [], "candidate_digest": head}
    if selector is not None:
        args["board"] = selector

    out = json.loads(registry.dispatch("kanban_review_findings", args))

    assert called == []
    assert out.get("ok") is not True
    assert out.get("error")
    assert kb.get_task(conn, tid).status == "running"


def test_conflicting_database_pin_cannot_mutate_another_board(board, monkeypatch):
    """With an omitted board selector the post-connection board check is skipped in
    the current code, so HERMES_KANBAN_DB repointed at a copy of the worker's own
    database (placed inside a foreign board's directory) can flip tasks to 'done' on
    that foreign copy, leaving the worker's own task 'running'. The fix must compare
    the connection's effective board against the validated HERMES_KANBAN_BOARD pin
    even when no explicit selector is supplied.
    """
    kb, conn, tid, head, review = board
    foreign = "copied-board"
    kb.create_board(foreign)
    dst_db = kb.board_dir(foreign) / "kanban.db"
    conn.commit()
    kb._INITIALIZED_PATHS.clear()
    # A live WAL database is not a consistent snapshot on disk: a plain file
    # copy takes the main file and leaves the -wal sidecar behind, so the rows
    # written by this fixture would be missing from the copy. SQLite's own
    # online-backup API produces a consistent copy including the WAL contents.
    import sqlite3

    backup_target = sqlite3.connect(dst_db)
    try:
        conn.backup(backup_target)
    finally:
        backup_target.close()

    # Prove the copy is a real, usable foreign board carrying this task; without
    # this the non-mutation assertion below could pass vacuously on an empty copy.
    seeded = kb.connect(db_path=dst_db)
    try:
        seeded_task = kb.get_task(seeded, tid)
        assert seeded_task is not None
        assert seeded_task.status == "running"
    finally:
        seeded.close()

    monkeypatch.setenv("HERMES_KANBAN_DB", str(dst_db))
    monkeypatch.setenv("HERMES_KANBAN_BOARD", kb.DEFAULT_BOARD)

    from tools.registry import registry

    out = json.loads(registry.dispatch("kanban_review_findings", {
        "task_id": tid,
        "findings": [],
        "candidate_digest": head,
    }))

    assert out.get("ok") is not True
    assert out.get("error")

    foreign_conn = kb.connect(db_path=dst_db)
    try:
        foreign_task = kb.get_task(foreign_conn, tid)
        assert foreign_task is not None
        assert foreign_task.status == "running"
    finally:
        foreign_conn.close()

    assert kb.get_task(conn, tid).status == "running"


@pytest.mark.parametrize("selector", [None, "default"])
def test_conflicting_database_pin_is_refused_before_any_connection(
    board, monkeypatch, selector
):
    """HERMES_KANBAN_DB outranks the board argument in the shared resolver, so a
    valid board pin plus a conflicting database pin is an inconsistent target; the
    connection must not be opened at all while the target is unproven, because every
    connection this helper opens is write-capable (it enables WAL and, on first open,
    runs schema creation and the additive migrations).

    Note that the sentinel stops the actual connection, so this pins down call
    ORDERING — it is not a claim that foreign writes were measured.
    """
    kb, conn, tid, head, review = board
    kb.create_board("foreign-review-board")
    import tools.kanban_tools as kt
    from tools.registry import registry

    # The trusted board pin stays VALID and unchanged; only the database pin is
    # repointed at another existing board's store.
    monkeypatch.setenv(
        "HERMES_KANBAN_DB",
        str(kb.board_dir("foreign-review-board") / "kanban.db"),
    )
    monkeypatch.setenv("HERMES_KANBAN_BOARD", kb.DEFAULT_BOARD)

    called = []

    def _sentinel_connect(*args, **kwargs):
        called.append((args, kwargs))
        raise AssertionError("write-capable connect reached before refusal")

    monkeypatch.setattr(kt, "_connect", _sentinel_connect)

    args = {"task_id": tid, "findings": [], "candidate_digest": head}
    if selector is not None:
        args["board"] = selector

    out = json.loads(registry.dispatch("kanban_review_findings", args))

    assert called == [], out
    assert out.get("ok") is not True
    assert out.get("error")
    assert kb.get_task(conn, tid).status == "running"


def test_omitted_board_with_valid_pin_still_opens_the_connection(board, monkeypatch):
    """With a valid HERMES_KANBAN_BOARD pin and no explicit board selector, the call
    must still reach _connect and succeed normally. This is the positive control for
    requirement (a): enforcing a valid pin must not block ordinary omitted-selector
    calls when the pin itself is well-formed.
    """
    kb, conn, tid, head, review = board
    import tools.kanban_tools as kt
    from tools.registry import registry

    real_connect = kt._connect
    called = []

    def _recording_connect(*args, **kwargs):
        called.append((args, kwargs))
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(kt, "_connect", _recording_connect)

    out = json.loads(registry.dispatch("kanban_review_findings", {
        "task_id": tid,
        "findings": [],
        "candidate_digest": head,
    }))

    assert called != []
    assert out.get("ok") is True, out
    assert out["outcome"] == "passed"

    task = kb.get_task(conn, tid)
    assert task.status == "done"


# ---------------------------------------------------------------------------
# 8. Delegated-child context -> denial
# ---------------------------------------------------------------------------


def test_delegated_child_context_is_denied(board):
    kb, conn, tid, head, review = board
    from agent.delegation_context import delegated_child_context

    with delegated_child_context():
        out = _dispatch({"findings": [_finding(candidate_digest=head)]})
    assert out.get("ok") is not True
    assert "delegate_task child" in out["error"]

    task = kb.get_task(conn, tid)
    assert task is not None
    assert task.status == "running"
    assert kb.list_attachments(conn, tid) == []


# ---------------------------------------------------------------------------
# Reviewer-run boundary: only the active reviewer run of THIS task may
# submit a verdict. Neither an absent run pin nor a stale/foreign one may
# fall back to any broader "orchestrator" allowance for this one tool.
# ---------------------------------------------------------------------------


def test_missing_run_id_is_refused_with_no_write(board, monkeypatch):
    """HERMES_KANBAN_RUN_ID absent, HERMES_KANBAN_TASK still pinned.

    ``_worker_run_id`` would return ``None`` here, and the kernel skips its
    run-match check entirely when ``expected_run_id`` is ``None`` — a
    reviewer-verdict authority hole. This caller must be refused outright,
    not treated as an orchestrator with unrestricted access to the task.
    """
    kb, conn, tid, head, review = board
    monkeypatch.delenv("HERMES_KANBAN_RUN_ID", raising=False)
    before = kb.get_task(conn, tid)

    out = _dispatch({"findings": [_finding(candidate_digest=head)]})
    assert out.get("ok") is not True
    assert out.get("error")
    assert "active reviewer run" in out["error"]

    after = kb.get_task(conn, tid)
    assert after.status == before.status
    assert after.status != "done"
    assert kb.list_attachments(conn, tid) == []


def test_missing_task_and_run_pins_with_kanban_toolset_profile_is_refused(
    board, tmp_path, monkeypatch
):
    """No worker pins at all, but the profile enables the ``kanban`` toolset.

    ``_enforce_worker_task_ownership`` deliberately allows this shape (an
    orchestrator profile with no ``HERMES_KANBAN_TASK`` routes work freely),
    but a review verdict is not a routing action — this caller must still
    be refused, explicit ``task_id`` and all.
    """
    kb, conn, tid, head, review = board
    profile_home = tmp_path / "profile-home"
    profile_home.mkdir()
    (profile_home / "config.yaml").write_text("toolsets:\n  - kanban\n")
    # Keep the board pins (so the tool still resolves to the right DB) but
    # drop the worker identity pins and repoint HERMES_HOME at a profile
    # home that independently enables the kanban toolset.
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_RUN_ID", raising=False)
    monkeypatch.setenv("HERMES_PROFILE", "reviewer-orchestrator")
    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    before = kb.get_task(conn, tid)

    out = _dispatch({"task_id": tid, "findings": [_finding(candidate_digest=head)]})
    assert out.get("ok") is not True
    assert out.get("error")
    assert "active reviewer run" in out["error"]

    after = kb.get_task(conn, tid)
    assert after.status == before.status
    assert after.status != "done"
    assert kb.list_attachments(conn, tid) == []


def test_delegated_child_context_refused_before_any_connection(board):
    """Same denial as the delegated-child test above, but pinned specifically
    to the reviewer-run gate: even with valid TASK/RUN_ID pins in the env,
    a delegate_task child must never reach the kernel."""
    kb, conn, tid, head, review = board
    from agent.delegation_context import delegated_child_context

    before = kb.get_task(conn, tid)
    with delegated_child_context():
        out = _dispatch({"findings": [_finding(candidate_digest=head)]})
    assert out.get("ok") is not True
    assert out.get("error")

    after = kb.get_task(conn, tid)
    assert after.status == before.status
    assert kb.list_attachments(conn, tid) == []


def test_in_process_cron_context_is_refused_with_no_write(board):
    """A cron job fired in-process from a worker inherits the worker's
    HERMES_KANBAN_TASK/RUN_ID env, but is not the dispatcher-owned worker
    itself and must not be able to submit that worker's review verdict.

    Passes ``task_id`` explicitly so ``_default_task_id``'s own cron guard
    (which only blocks the *implicit* task-id fallback) is not what's under
    test here — this pins down the reviewer-run gate specifically: neither
    ``_worker_run_id`` nor ``_enforce_worker_task_ownership`` check dispatcher
    ownership, so an in-process cron job naming its host worker's task
    explicitly could reach the kernel before this gate existed.
    """
    kb, conn, tid, head, review = board
    from agent.delegation_context import non_dispatcher_owned_context

    before = kb.get_task(conn, tid)
    with non_dispatcher_owned_context():
        out = _dispatch({
            "task_id": tid,
            "findings": [_finding(candidate_digest=head)],
        })
    assert out.get("ok") is not True
    assert out.get("error")
    assert "active reviewer run" in out["error"]

    after = kb.get_task(conn, tid)
    assert after.status == before.status
    assert kb.list_attachments(conn, tid) == []


# ---------------------------------------------------------------------------
# 9. Registration + toolset exposure
# ---------------------------------------------------------------------------


def test_registered_and_exposed_in_both_toolset_lists():
    import tools.kanban_tools  # noqa: F401 - ensure registered
    from tools.registry import registry
    from toolsets import TOOLSETS, _HERMES_CORE_TOOLS

    entry = registry.get_entry("kanban_review_findings")
    assert entry is not None
    assert entry.toolset == "kanban"
    assert callable(entry.handler)
    assert entry.schema["name"] == "kanban_review_findings"
    assert entry.schema["parameters"]["required"] == ["findings"]

    assert "kanban_review_findings" in _HERMES_CORE_TOOLS
    assert "kanban_review_findings" in TOOLSETS["kanban"]["tools"]


def test_database_pin_is_not_mutated_during_a_successful_call(board, monkeypatch):
    """An authority check must not make the trusted pin momentarily absent,
    because concurrent readers in the same process would resolve a lower-precedence
    database during that window when HERMES_KANBAN_DB is transiently removed.
    """
    kb, conn, tid, head, review = board
    from tools.registry import registry

    original_pin = os.environ["HERMES_KANBAN_DB"]
    observed_pins = []
    real_kanban_db_path = kb.kanban_db_path

    def _observing_kanban_db_path(*args, **kwargs):
        observed_pins.append(os.environ.get("HERMES_KANBAN_DB"))
        return real_kanban_db_path(*args, **kwargs)

    monkeypatch.setattr(kb, "kanban_db_path", _observing_kanban_db_path)

    out = json.loads(registry.dispatch("kanban_review_findings", {
        "task_id": tid,
        "findings": [],
        "candidate_digest": head,
    }))

    assert out.get("ok") is True, out
    assert out["outcome"] == "passed"
    assert len(observed_pins) >= 1, "observer recorded no calls to kanban_db_path"
    assert all(v == original_pin for v in observed_pins), (
        f"HERMES_KANBAN_DB was transiently mutated during a successful call: {observed_pins!r}"
    )
    assert os.environ["HERMES_KANBAN_DB"] == original_pin


# ---------------------------------------------------------------------------
# Schema exposure: only the run claimed from the review lane lists the tool
# ---------------------------------------------------------------------------

def _schema_tool_names() -> set:
    from tools.registry import invalidate_check_fn_cache, registry
    from toolsets import resolve_toolset

    invalidate_check_fn_cache()
    schema = registry.get_definitions(set(resolve_toolset("hermes-cli")), quiet=True)
    return {s["function"].get("name") for s in schema if "function" in s}


def test_review_lane_run_lists_the_tool(board):
    assert "kanban_review_findings" in _schema_tool_names()


def test_worker_run_not_claimed_from_review_does_not_list_the_tool(board, monkeypatch):
    kb, conn, _tid, _head, _review = board
    other = kb.create_task(conn, title="ordinary build", assignee=IMPLEMENTER)
    run = kb.claim_task(conn, other, claimer=f"{IMPLEMENTER}:2")
    assert run is not None
    monkeypatch.setenv("HERMES_KANBAN_TASK", other)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run.current_run_id))

    names = _schema_tool_names()
    assert "kanban_complete" in names
    assert "kanban_review_findings" not in names


def test_kanban_toolset_profile_without_task_pin_does_not_list_the_tool(
    board, tmp_path, monkeypatch,
):
    (tmp_path / ".hermes" / "config.yaml").write_text(
        "toolsets:\n  - kanban\n", encoding="utf-8",
    )
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_RUN_ID", raising=False)

    names = _schema_tool_names()
    assert "kanban_show" in names
    assert "kanban_review_findings" not in names
