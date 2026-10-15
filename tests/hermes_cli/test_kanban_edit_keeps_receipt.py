"""A later result edit keeps the platform-written receipts (finding 78).

``edit_completed_task_result`` wrote the edited metadata over the completed
run's whole metadata. One edit therefore erased, or replaced with whatever the
caller sent, the keys only the platform writes: the session link
(``worker_session_id``), the model and capability receipt
(``runtime_receipt``) and both code-change receipts (``execution_receipt``
and ``worktree_materialization``).

Owner decision of 2026-10-15: a result edit never rewrites those receipts. An
edit changes only the ordinary keys, and a run the edit itself has to create
carries none of the platform keys. An edit without metadata leaves the run's
metadata exactly as it was, and the edit event keeps its shape.

Stored metadata is read back from the board file with plain SQL, so
"unchanged" here means the stored bytes of each receipt.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb

WORKER = "builder"
SESSION = "builder-session"
BRANCH = "feature/edited-result"
OWNED = "src/owned"
PATCHED = f"{OWNED}/feature.py"
SEED_AUTHOR = ("Board Owner", "owner@example.invalid")
_GIT_ENV = {
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_SYSTEM": os.devnull,
    "GIT_TERMINAL_PROMPT": "0",
}

# The keys only the platform writes, in the order the owner decision names them.
PLATFORM_KEYS = (
    "runtime_receipt",
    "worker_session_id",
    "execution_receipt",
    "worktree_materialization",
)

# What an edit might send under those keys.
MADE_UP = {
    "runtime_receipt": {
        "schema_version": 3,
        "engine": "hermes",
        "profile": "made-up-profile",
        "provider": "made-up-provider",
        "model": "made-up-model",
        "reasoning_effort": "high",
        "route_evidence": "session-row",
        "capability": {"tools": ["made_up_tool"], "source": "session-tool-calls"},
    },
    "worker_session_id": "made-up-session",
    "execution_receipt": {"head_commit": "0" * 40, "changed_paths": ["made/up.py"]},
    "worktree_materialization": {
        "patch_attachment_id": 999,
        "materialized_head": "0" * 40,
        "patch_sha256": "0" * 64,
    },
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


def _record_session(state_db: Path) -> None:
    """The worker's session log in its profile's real session store."""
    from hermes_state import SessionDB

    state_db.parent.mkdir(parents=True, exist_ok=True)
    db = SessionDB(db_path=state_db)
    try:
        db.create_session(SESSION, source="kanban", model="requested-model", profile_name=WORKER)
        for index, name in enumerate(("read_file", "terminal")):
            call_id = f"call-{index}"
            db.append_message(
                SESSION, "assistant", content="",
                tool_calls=[{
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": "{}"},
                }],
            )
            db.append_message(
                SESSION, "tool", content="done", tool_name=name, tool_call_id=call_id,
            )
        db.update_token_counts(
            SESSION, input_tokens=10, output_tokens=4,
            model="served-model", billing_provider="served-provider",
            estimated_cost_usd=0.01, cost_status="estimated",
            cost_source="official_docs_snapshot", api_call_count=1,
        )
    finally:
        db.close()


def _stored_metadata(conn, run_id: int):
    """The run's metadata exactly as stored, read with plain SQL."""
    row = conn.execute(
        "SELECT metadata FROM task_runs WHERE id = ?", (run_id,)
    ).fetchone()
    assert row is not None, f"run {run_id} is missing"
    return row["metadata"]


def _last_edit_event(conn, task_id: str):
    edits = [event for event in kb.list_events(conn, task_id) if event.kind == "edited"]
    assert edits, "the edit recorded no event"
    return edits[-1]


@pytest.fixture
def board(tmp_path, monkeypatch):
    """The default board of a temporary Hermes root, opened as its owner."""
    home = tmp_path / "hermes_home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
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

    kb.init_db()
    conn = kb.connect()
    try:
        yield conn
    finally:
        conn.close()


@pytest.fixture
def completed_build(board, tmp_path):
    """A scoped build the kernel completed, carrying all four platform receipts.

    The worker's heartbeat linked the run to its session, the kernel applied
    the worker's patch inside the owned path, and the kernel's own run ending
    stamped the model and capability receipt from the session store.
    """
    from hermes_cli.profiles import get_profile_dir

    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "--initial-branch=main")
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "init", author=SEED_AUTHOR)

    conn = board
    task_id = kb.create_task(
        conn, title="scoped build", assignee=WORKER,
        workspace_kind="worktree", workspace_path=str(repo),
        branch_name=BRANCH, owned_paths=[OWNED],
    )
    claimed = kb.claim_task(conn, task_id)
    assert claimed is not None and claimed.current_run_id is not None
    run_id = claimed.current_run_id
    workspace, branch = kb._resolve_worktree_workspace(claimed)
    kb.set_workspace_path(conn, task_id, str(workspace))
    kb.set_branch_name(conn, task_id, branch)
    kb.record_worktree_base(conn, task_id, workspace)

    _record_session(get_profile_dir(WORKER) / "state.db")
    assert kb.heartbeat_worker(conn, task_id, expected_run_id=run_id, session_id=SESSION)
    attachment_id = kb.store_attachment_bytes(
        conn, task_id, "sandbox-result.patch",
        _new_file_patch(PATCHED, "feature = True\n"),
        content_type="text/x-diff", uploaded_by="agent",
    )
    assert kb.complete_task(
        conn, task_id,
        summary="built the feature",
        metadata={"tests_run": 3},
        patch_attachment_id=attachment_id,
        expected_run_id=run_id,
    )

    stored = json.loads(_stored_metadata(conn, run_id))
    assert set(PLATFORM_KEYS) <= set(stored), sorted(stored)
    assert stored["worker_session_id"] == SESSION
    assert stored["runtime_receipt"]["model"] == "served-model"
    assert stored["runtime_receipt"]["capability"]["tools"] == ["read_file", "terminal"]
    assert stored["execution_receipt"]["changed_paths"] == [PATCHED]
    assert stored["worktree_materialization"]["patch_attachment_id"] == attachment_id
    assert stored["tests_run"] == 3
    return {"conn": conn, "task": task_id, "run": run_id}


def test_an_edit_with_a_made_up_receipt_keeps_the_platform_receipts(completed_build):
    """Test 4: the original receipt and session link stay; the new ordinary key is added."""
    conn, task_id, run_id = (
        completed_build["conn"], completed_build["task"], completed_build["run"],
    )
    before_raw = _stored_metadata(conn, run_id)
    before = json.loads(before_raw)

    edited = {**MADE_UP, "reviewer_note": "result corrected after review"}
    assert kb.edit_completed_task_result(
        conn, task_id,
        result="corrected result",
        summary="corrected summary",
        metadata=edited,
    )

    event = _last_edit_event(conn, task_id)
    assert event.run_id == run_id
    assert event.payload == {
        "fields": ["result", "summary", "metadata"],
        "result_len": len("corrected result"),
        "summary": "corrected summary",
    }

    after_raw = _stored_metadata(conn, run_id)
    after = json.loads(after_raw)
    for key in PLATFORM_KEYS:
        # The receipt's own stored bytes, as the kernel wrote them.
        member = json.dumps({key: before[key]}, ensure_ascii=False)[1:-1]
        assert member in before_raw, key
        assert member in after_raw, f"the edit rewrote {key}: {after.get(key)!r}"
        assert after[key] == before[key], key
    assert after["reviewer_note"] == "result corrected after review"
    assert after["tests_run"] == 3
    assert set(after) == set(before) | {"reviewer_note"}


def test_an_edit_that_creates_the_run_stores_no_made_up_receipt(board):
    """Test 5: a done card with no completed run gets one, without the made-up receipt."""
    conn = board
    task_id = kb.create_task(conn, title="marked done by the owner", assignee=WORKER)
    status = kb.get_task(conn, task_id).status
    moved = kb.cas_transition_task(
        conn, task_id,
        expected_status=status,
        expected_revision=kb.task_event_revision(conn, task_id),
        to_status="done",
        event_kind="owner_move",
        event_payload={"to": "done"},
    )
    assert moved["moved"] is True
    assert conn.execute(
        "SELECT COUNT(*) FROM task_runs WHERE task_id = ? AND outcome = 'completed'",
        (task_id,),
    ).fetchone()[0] == 0

    edited = {**MADE_UP, "note": "result recorded by the owner"}
    assert kb.edit_completed_task_result(conn, task_id, result="the result", metadata=edited)

    runs = conn.execute(
        "SELECT id, outcome, metadata FROM task_runs WHERE task_id = ?", (task_id,)
    ).fetchall()
    assert len(runs) == 1 and runs[0]["outcome"] == "completed"
    assert _last_edit_event(conn, task_id).run_id == runs[0]["id"]
    stored = json.loads(runs[0]["metadata"])
    assert "runtime_receipt" not in stored, stored["runtime_receipt"]
    for key in PLATFORM_KEYS:
        assert key not in stored, key
    assert stored == {"note": "result recorded by the owner"}


def test_an_edit_without_metadata_leaves_the_run_metadata_byte_for_byte(completed_build):
    """Check 7: no metadata in the edit, no change to the stored metadata."""
    conn, task_id, run_id = (
        completed_build["conn"], completed_build["task"], completed_build["run"],
    )
    before_raw = _stored_metadata(conn, run_id)

    assert kb.edit_completed_task_result(conn, task_id, result="corrected result")

    assert _stored_metadata(conn, run_id) == before_raw
    event = _last_edit_event(conn, task_id)
    assert event.run_id == run_id
    assert event.payload["fields"] == ["result", "summary"]
