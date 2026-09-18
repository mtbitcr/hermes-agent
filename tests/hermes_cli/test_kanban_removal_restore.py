"""§9.4: a validated restore, paused under a new epoch, that cannot be raced.

One regression per point the restore owes, plus the three cases named
explicitly: a TAMPERED retained set, a STALE HANDLE after the restore, and
a CRASH MID-RESTORE.

Durable facts are read through a plain read-only connection, never through
the module under test, and the crash is a real SIGKILL in a real subprocess
driving the shipped CLI.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

_WORKTREE = Path(__file__).resolve().parents[2]
if str(_WORKTREE) not in sys.path:
    sys.path.insert(0, str(_WORKTREE))

from hermes_cli import kanban_db as kb
from tests.hermes_cli._kanban_fence_support import (
    add_linked_work_area, cli, create_fenced_board, git, make_git_repo,
    read_only, ready_task, record_git_receipt, register_row, start_removal,
)

_CHILD = Path(__file__).parent / "_kanban_removal_crash_child.py"
_CLAIM_CHILD = Path(__file__).parent / "_kanban_restore_claim_child.py"
#: Dies with the retained set installed and the new epoch not yet frozen.
_MID_RESTORE = "restore-after-install"


def _child_env(home: Path) -> dict:
    return {"PATH": "/usr/bin:/bin:/usr/local/bin", "PYTHONPATH": str(_WORKTREE),
            "PYTHONHASHSEED": "0", "TZ": "UTC", "LANG": "C.UTF-8",
            "HOME": str(home)}


def _run_child(home: Path, slug: str, crash: str, mode: str):
    return subprocess.run(
        [sys.executable, str(_CHILD), str(home), slug, crash, mode],
        capture_output=True, text=True, timeout=180, cwd=str(_WORKTREE),
        env=_child_env(home),
    )


def _claim_and_depart(home: Path, slug: str, task: str) -> str:
    """A REAL worker process claims *task*, binds its pid, and exits.

    Returns the claim handle it minted. The process is gone by the time
    this returns, so the removal's own quiescence can read the holder as
    PROVABLY absent — the only way a reversible removal ever completes
    over a claim that was taken before it started.
    """
    done = subprocess.run(
        [sys.executable, str(_CLAIM_CHILD), str(home), slug, task],
        capture_output=True, text=True, timeout=180, cwd=str(_WORKTREE),
        env=_child_env(home),
    )
    assert done.returncode == 0, f"out={done.stdout} err={done.stderr}"
    line, = [ln for ln in done.stdout.splitlines() if ln.startswith("CLAIM ")]
    minted = json.loads(line[len("CLAIM "):])
    # The row really is Held by that departed process, with its identity on it.
    with read_only(kb.kanban_db_path(board=slug)) as conn:
        row = conn.execute(
            "SELECT status, claim_lock, worker_pid FROM tasks WHERE id = ?", (task,)
        ).fetchone()
    assert row["status"] == "running"
    assert row["claim_lock"] == minted["claim_lock"]
    assert row["worker_pid"] == minted["pid"]
    return minted["claim_lock"]


def _tasks(db_path: Path) -> list:
    with read_only(db_path) as conn:
        return sorted(row["id"] for row in conn.execute("SELECT id FROM tasks"))


def _row(slug: str):
    with read_only(kb.register_db_path()) as conn:
        return conn.execute(
            "SELECT phase, outcome, removal_id, apply_journal FROM "
            "board_removal_phase WHERE board_name = ?", (slug,)).fetchone()


def _journal(slug: str) -> list:
    raw = _row(slug)["apply_journal"]
    return json.loads(raw)["items"] if raw else []


def _steps(slug: str, *, ok: bool = True) -> set:
    return {
        item["step"] for item in _journal(slug)
        if bool(item.get("ok")) is ok and item.get("step")
    }


def _gate(slug: str):
    """``(gate, epoch_mirror)`` read straight off the restored store."""
    with read_only(kb.kanban_db_path(board=slug)) as conn:
        row = conn.execute(
            "SELECT gate, epoch_mirror FROM board_fence_state WHERE id = 1"
        ).fetchone()
    return None if row is None else (row["gate"], int(row["epoch_mirror"]))


def _archived(
    slug: str, *, departed_holder: bool = False, repo: Path = None
) -> dict:
    """A REAL reversible removal driven to done, with real work in it.

    ``departed_holder`` has a real child process claim the task and exit
    before the removal starts, so the handle that claim minted is a real
    pre-removal handle — and the removal completes only because its own
    quiescence can prove the holder gone.
    """
    create_fenced_board(slug)
    conn = kb.connect(board=slug)
    task = ready_task(conn, title="work an archive has to keep")
    if repo is not None:
        base = make_git_repo(repo)
        work_area, head = add_linked_work_area(repo, task, branch=f"hermes/{task}")
        record_git_receipt(conn, task, workspace_path=work_area,
                           branch_name=f"hermes/{task}", base_commit=base,
                           head_commit=head)
    conn.close()
    tasks = _tasks(kb.kanban_db_path(board=slug))
    claim = (
        _claim_and_depart(Path(os.environ["HERMES_HOME"]), slug, task)
        if departed_holder else None
    )

    result = kb.remove_board_fenced(slug, mode=kb.RemovalMode.REVERSIBLE)
    assert result.success, result.message
    row = _row(slug)
    assert (row["phase"], row["outcome"]) == ("done", "archived")
    return {
        "slug": slug, "task": task, "tasks": tasks,
        "removal_id": row["removal_id"], "claim": claim,
        "retained": kb.reversible_retained_path(slug, row["removal_id"]),
    }


# ---------------------------------------------------------------------------
# 1. The complete retained set is validated before any live write
# ---------------------------------------------------------------------------

def test_a_tampered_retained_set_is_refused_and_nothing_live_is_written(
    fence_home
):
    """Three ways a set is not the set that was archived, each refused."""
    changed = _archived("restore-tampered")
    truncated = _archived("restore-short")
    extra = _archived("restore-extra")

    # (a) A file whose content is not what was recorded.
    store = changed["retained"] / "kanban.db"
    store.write_bytes(store.read_bytes()[:-1] + b"X")
    # (b) A file whose size is not what was recorded.
    (truncated["retained"] / "board.json").write_text("{}", encoding="utf-8")
    # (c) A file the recorded set does not name at all.
    (extra["retained"] / "smuggled.txt").write_text("extra", encoding="utf-8")

    for case, needle in (
        (changed, "content of"), (truncated, "byte(s) where"),
        (extra, "does not record"),
    ):
        slug = case["slug"]
        result = kb.restore_retained_board(slug, removal_id=case["removal_id"])

        assert result.success is False, result.message
        assert "TAMPERED" in result.message and needle in result.message
        assert result.state == kb.RESTORE_STATE_REFUSED
        # Nothing live was touched: no store, no directory, and the
        # register still records the board archived.
        assert not kb.kanban_db_path(board=slug).exists()
        assert not kb.board_dir(slug).exists()
        assert register_row(slug)["lifecycle"] == kb.BoardLifecycle.ARCHIVED.value
        assert kb.RESTORE_JOURNAL_INSTALLED not in _steps(slug)
        assert kb.restore_status(
            kb.get_removal_phase_record(slug)
        )["state"] == kb.RESTORE_STATE_REFUSED


def test_an_incomplete_retained_set_is_refused_with_a_plain_reason(fence_home):
    """A missing member of the recorded set refuses, naming the file."""
    case = _archived("restore-incomplete")
    (case["retained"] / "board.json").unlink()

    result = kb.restore_retained_board(case["slug"])

    assert result.success is False
    assert "INCOMPLETE" in result.message and "board.json" in result.message
    assert not kb.kanban_db_path(board=case["slug"]).exists()
    assert kb.RESTORE_JOURNAL_VALIDATED not in _steps(case["slug"])


# ---------------------------------------------------------------------------
# 2. A successful restore returns the board PAUSED under a NEW epoch, and
#    every handle from before the removal is void
# ---------------------------------------------------------------------------

def test_a_stale_handle_cannot_act_on_the_restored_board(fence_home):
    """Restored paused at a new epoch; the pre-removal handle cannot act.

    The handle is a real one: a child process claimed the task, bound its
    own pid to the claim and exited, which is the ONLY way a reversible
    removal ever completes over a prior claim — its quiescence has to read
    the holder as provably absent.
    """
    case = _archived("restore-paused", departed_holder=True)
    slug, task = case["slug"], case["task"]
    stale = case["claim"]
    assert stale is not None
    epoch_before = register_row(slug)["epoch"]

    result = kb.restore_retained_board(slug, removal_id=case["removal_id"])

    assert result.success, result.message
    assert result.state == kb.RESTORE_STATE_RESTORED
    # A NEW epoch, STRICTLY past the archived one, on the register and
    # mirrored in the restored store, which is FROZEN at it.
    assert result.epoch > epoch_before
    entry = kb.get_register_entry(slug)
    assert (entry.lifecycle, entry.epoch) == (kb.BoardLifecycle.LIVE, result.epoch)
    assert _gate(slug) == ("frozen", result.epoch)
    # The board's content really came back.
    assert _tasks(kb.kanban_db_path(board=slug)) == case["tasks"]

    # PAUSED: the fence refuses ordinary work until an explicit resume.
    conn = kb.connect(board=slug)
    try:
        with pytest.raises(kb.BoardFenceClosedError):
            kb.claim_task(conn, task)
    finally:
        conn.close()

    resumed = kb.resume_restored_board(slug)
    assert resumed.success, resumed.message
    # Open at the NEW epoch, and the register and the in-board mirror agree
    # on it — the restored board carries one epoch, not two accounts of it.
    assert _gate(slug) == ("open", result.epoch)
    assert kb.get_register_entry(slug).epoch == result.epoch

    conn = kb.connect(board=slug)
    try:
        # The claim the departed holder minted is not on the restored row,
        # so the compare-and-set that would extend it cannot match.
        assert kb.heartbeat_claim(conn, task, claimer=stale) is False
        with read_only(kb.kanban_db_path(board=slug)) as ro:
            row = ro.execute(
                "SELECT status, claim_lock FROM tasks WHERE id = ?", (task,)
            ).fetchone()
        assert row["status"] == "ready" and row["claim_lock"] is None
        # And the board really is usable again for NEW work.
        assert kb.claim_task(conn, task) is not None
    finally:
        conn.close()


def _rerecord_retained_file(slug: str, relative: str) -> None:
    """Re-record *relative*'s REAL size and digest in the retained metadata.

    §7.3 captured the manifest as it left the retained copy, so a test that
    edits that copy has to leave the recorded sizes and digests describing
    the bytes actually on disk — otherwise validation refuses the set as
    TAMPERED and the restore never reaches the thing under test. This
    rewrites the recorded pair and nothing else: the manifest stays the
    authority the restore validates against.
    """
    row = _row(slug)
    payload = json.loads(row["apply_journal"])
    archived = [
        item for item in payload["items"]
        if item.get("action") == kb.APPLY_JOURNAL_RETAINED_ARCHIVED
        and item.get("ok")
    ]
    assert archived, "the removal recorded no archived retained copy"
    target = kb.reversible_retained_path(slug, row["removal_id"]) / relative
    entry, = [
        member for member in archived[-1]["detail"]["manifest"]["files"]
        if member["path"] == relative
    ]
    entry["size"] = target.stat().st_size
    entry["sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
    conn = sqlite3.connect(kb.register_db_path())
    try:
        conn.execute(
            "UPDATE board_removal_phase SET apply_journal = ? WHERE board_name = ?",
            (json.dumps(payload), slug),
        )
        conn.commit()
    finally:
        conn.close()


def test_a_claim_inside_the_retained_set_is_voided_by_the_restore(fence_home):
    """A retained copy that DOES hold a claim comes back with it voided.

    The restore's pause step is what makes "every handle recorded before
    the removal is void" true of the rows themselves, not only of the
    gate. This drives that step over a retained set which genuinely
    contains a running, claimed row — recorded in the manifest with its
    real size and digest, so the set still validates — and reads the
    voiding back off the restored store and off the durable journal.
    """
    case = _archived("restore-claimed")
    slug, task = case["slug"], case["task"]
    store = case["retained"] / "kanban.db"
    before = sorted(p.name for p in case["retained"].iterdir())

    # The retained COPY (the live board is gone) is made to hold a real
    # reservation: a running row with a claim, an expiry and a worker on it.
    held = sqlite3.connect(store)
    try:
        held.execute(
            "UPDATE tasks SET status = 'running', claim_lock = ?, "
            "claim_expires = ?, worker_pid = ? WHERE id = ?",
            ("archived-host:archived-worker", 2_000_000_000, 4242, task),
        )
        held.commit()
    finally:
        held.close()
    assert sorted(p.name for p in case["retained"].iterdir()) == before, (
        "editing the retained store left files the manifest does not record"
    )
    _rerecord_retained_file(slug, "kanban.db")

    # The set validates: the bytes on disk are the bytes recorded for it.
    manifest, reason = kb._recorded_retained_manifest(kb.get_removal_phase_record(slug))
    valid, valid_reason, _ = kb.validate_retained_set(case["retained"], manifest)
    assert valid, f"{reason}: {valid_reason}"
    with read_only(store) as ro:
        row = ro.execute(
            "SELECT status, claim_lock FROM tasks WHERE id = ?", (task,)
        ).fetchone()
    assert (row["status"], row["claim_lock"]) == (
        "running", "archived-host:archived-worker"
    )

    result = kb.restore_retained_board(slug, removal_id=case["removal_id"])

    assert result.success, result.message
    # The reservation did not come back with the board.
    with read_only(kb.kanban_db_path(board=slug)) as ro:
        restored = ro.execute(
            "SELECT status, claim_lock, claim_expires, worker_pid, current_run_id "
            "FROM tasks WHERE id = ?", (task,)
        ).fetchone()
        kinds = [
            event["kind"] for event in ro.execute(
                "SELECT kind FROM task_events WHERE task_id = ?", (task,)
            )
        ]
    assert restored["status"] == "ready"
    assert restored["claim_lock"] is None
    assert restored["claim_expires"] is None
    assert restored["worker_pid"] is None
    assert restored["current_run_id"] is None
    # And the voiding is RECORDED: on the board, as the event the restore
    # emits, and on the removal's own journal as what it voided.
    assert "restore_reservation_voided" in kinds
    completed = [
        item for item in _journal(slug)
        if item.get("action") == kb.RESTORE_JOURNAL_COMPLETED and item.get("ok")
    ]
    assert completed, "the restore recorded no completion"
    assert completed[-1]["detail"]["voided"] == [task]
    assert "1 reservation(s) recorded before the removal were voided" in (
        completed[-1]["reason"]
    )

    # The handle that claim named can do nothing on the restored board.
    assert kb.resume_restored_board(slug).success
    conn = kb.connect(board=slug)
    try:
        assert kb.heartbeat_claim(
            conn, task, claimer="archived-host:archived-worker"
        ) is False
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 3. Shared git state is never rewound; reattachment is provenance-scoped
# ---------------------------------------------------------------------------

def test_the_restore_never_rewinds_shared_git_state(fence_home, tmp_path):
    """Branches, commits and registrations stay exactly as they are.

    What the provenance ledger records as this board's own is attributed;
    a commit it does not record is reported UNVERIFIED and left alone.
    """
    repo = tmp_path / "shared-repo"
    case = _archived("restore-git", repo=repo)
    slug, task = case["slug"], case["task"]
    branch = f"hermes/{task}"
    head = git(repo, "rev-parse", branch)
    branches = git(repo, "branch", "--format=%(refname:short)")
    registrations = sorted(
        p.name for p in (repo / ".git" / "worktrees").iterdir()
    )

    result = kb.restore_retained_board(slug, removal_id=case["removal_id"])

    assert result.success, result.message
    # Nothing in the shared container moved.
    assert git(repo, "rev-parse", branch) == head
    assert git(repo, "branch", "--format=%(refname:short)") == branches
    assert sorted(
        p.name for p in (repo / ".git" / "worktrees").iterdir()
    ) == registrations

    report = result.reattachment
    assert report["acted_on"] == []
    assert report["source"] == kb.KERNEL_COMMIT_PROVENANCE_SOURCE
    # The commit this board's work produced has no creation-time ledger
    # row (nothing recorded one), so it is UNVERIFIED rather than claimed.
    unverified = {
        item.get("commit") for item in report["unverified"]
        if item["member"] == "version-control-reference"
    }
    assert head in unverified
    assert all(
        item["verdict"] == kb.RECEIPT_VERDICT_UNVERIFIED
        for item in report["unverified"]
    )

    # A commit the ledger DOES record as this board's own is attributed.
    kb.record_kernel_creation(
        family=kb.KERNEL_FAMILY_COMMIT, kind="worktree-advance",
        subject_id=head, board=slug, task_id=task, reference=branch,
    )
    again = kb.build_restore_reattachment(slug, epoch=result.epoch)
    assert head in {
        item.get("commit") for item in again["attributed"]
    }
    assert again["acted_on"] == []


# ---------------------------------------------------------------------------
# 4. Idempotent and crash-safe on the EXISTING phase record and receipt seam
# ---------------------------------------------------------------------------

def test_a_crash_mid_restore_completes_the_same_restore_never_a_second(
    fence_home
):
    """SIGKILL after the install: the restart finishes THIS restore."""
    case = _archived("restore-crash")
    slug = case["slug"]
    archived_epoch = register_row(slug)["epoch"]

    crashed = _run_child(fence_home, slug, _MID_RESTORE, "restore")

    assert crashed.returncode == -signal.SIGKILL, crashed.stderr
    # The new epoch was DECIDED and recorded before it was used, and the
    # board is not usable: no false "restored" in the middle.
    record = kb.get_removal_phase_record(slug)
    status = kb.restore_status(record)
    assert status["state"] == kb.RESTORE_STATE_RESTORING
    assert status["paused"] is True
    decided = status["epoch"]
    assert decided == archived_epoch + 1
    assert register_row(slug)["lifecycle"] == kb.BoardLifecycle.ARCHIVED.value
    assert kb.RESTORE_JOURNAL_COMPLETED not in _steps(slug)

    restarted = _run_child(fence_home, slug, "never", "restore")

    assert restarted.returncode == 0, restarted.stderr
    record = kb.get_removal_phase_record(slug)
    status = kb.restore_status(record)
    assert status["state"] == kb.RESTORE_STATE_RESTORED
    # The SAME restore: the epoch the crashed attempt recorded, not a new one.
    assert status["epoch"] == decided
    entry = kb.get_register_entry(slug)
    assert (entry.lifecycle, entry.epoch) == (kb.BoardLifecycle.LIVE, decided)
    assert _gate(slug) == ("frozen", decided)
    assert _tasks(kb.kanban_db_path(board=slug)) == case["tasks"]
    # ONE restore, recorded once, on the removal's OWN record — no second
    # phase record and no second state machine.
    assert [item["step"] for item in _journal(slug)].count(
        kb.RESTORE_JOURNAL_COMPLETED
    ) == 1
    with read_only(kb.register_db_path()) as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM board_removal_phase WHERE board_name = ?",
            (slug,),
        ).fetchone()[0] == 1
    assert _row(slug)["phase"] == "done"


def test_restore_is_idempotent_and_the_status_read_is_truthful(fence_home):
    """A second restore is a recorded no-op; the CLI reports the real state."""
    case = _archived("restore-idempotent")
    slug = case["slug"]

    first = kb.restore_retained_board(slug)
    assert first.success, first.message

    second = kb.restore_retained_board(slug)
    assert second.success and second.already_done
    assert second.epoch == first.epoch
    assert [item["step"] for item in _journal(slug)].count(
        kb.RESTORE_JOURNAL_COMPLETED
    ) == 1

    out = cli(f"boards removal-phase {slug} --json")
    payload = json.loads(out[out.index("{"):])
    assert payload["restore"]["state"] == kb.RESTORE_STATE_RESTORED
    assert payload["restore"]["paused"] is True
    assert payload["restore"]["epoch"] == first.epoch

    assert kb.resume_restored_board(slug).success
    out = cli(f"boards removal-phase {slug} --json")
    payload = json.loads(out[out.index("{"):])
    assert payload["restore"]["state"] == kb.RESTORE_STATE_RESUMED
    assert payload["restore"]["paused"] is False
    # A resume repeated is a no-op, not a second open.
    again = kb.resume_restored_board(slug)
    assert again.success and again.already_done


def test_a_board_that_is_present_is_never_written_over(fence_home):
    """A restore refuses rather than overwrite live storage."""
    case = _archived("restore-occupied")
    slug = case["slug"]
    # Something is live at that name again — whatever put it there, the
    # restore is not entitled to write over it.
    shutil.copytree(case["retained"], kb.board_dir(slug))
    marker = kb.board_dir(slug) / "kanban.db"
    marker.write_bytes(b"a live store this restore must not clobber")

    result = kb.restore_retained_board(slug)

    assert result.success is False
    assert "never writes over a board that is present" in result.message
    assert marker.read_bytes() == b"a live store this restore must not clobber"
    assert kb.restore_status(
        kb.get_removal_phase_record(slug)
    )["state"] == kb.RESTORE_STATE_REFUSED
