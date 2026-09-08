"""Shared helpers for the board-removal-fence regression suites.

Everything here either drives a REAL production API or reads persisted
state with a plain, read-only SQLite connection. Deliberately absent:
any helper that arms a board, opens a store, or mutates one by a route
production code does not take. The first fence repair passed its whole
suite while every shipped path stayed dormant precisely because its
tests started from helpers instead of from ``create_board`` /
``connect`` / ``claim_task``. :func:`create_fenced_board` is not an
exception to that: it is exactly ``create_board`` followed by the real
``backfill_register_entry`` an operator would run, composed once because
``create_board`` no longer arms a fence on its own.

The two exceptions that DO prove the rule and are named for it:
:func:`make_legacy_board` manufactures a board that PREDATES the fence
(the state the GA-5 backfill exists to migrate), and
:func:`hold_exclusive_lock` holds a real cross-process SQLite lock so the
mutation deadline can be measured against a real blocking read.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

_WORKTREE = Path(__file__).resolve().parents[2]
if str(_WORKTREE) not in sys.path:
    sys.path.insert(0, str(_WORKTREE))

from hermes_cli import kanban_db as kb  # noqa: E402


# ---------------------------------------------------------------------------
# Persisted-state readers (never through kanban_db)
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def read_only(db_path: Path):
    """A plain read-only connection — no kanban_db code in the path.

    Assertions about what is ON DISK must not be mediated by the module
    under test, or a fence that lies about its own state reads as green.
    """
    conn = sqlite3.connect(f"{Path(db_path).resolve().as_uri()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def table_names(db_path: Path) -> set:
    with read_only(db_path) as conn:
        return {
            r[0]
            for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }


def has_gate_table(db_path: Path) -> bool:
    return "board_fence_state" in table_names(db_path)


def gate_row(db_path: Path):
    """``(gate, epoch_mirror)`` as persisted, or None when Gate B is absent."""
    if not has_gate_table(db_path):
        return None
    with read_only(db_path) as conn:
        row = conn.execute(
            "SELECT gate, epoch_mirror FROM board_fence_state WHERE id = 1"
        ).fetchone()
    return None if row is None else (row["gate"], int(row["epoch_mirror"]))


def register_row(slug: str):
    """The board's Gate A row as persisted, or None."""
    path = kb.register_db_path()
    if not path.exists():
        return None
    with read_only(path) as conn:
        try:
            row = conn.execute(
                "SELECT lifecycle, epoch, epoch_before, gate_move "
                "FROM board_register WHERE board_name = ?",
                (slug,),
            ).fetchone()
        except sqlite3.Error:
            return None
    return None if row is None else dict(row)


def marker_row(slug: str) -> bool:
    path = kb.archive_db_path()
    if not path.exists():
        return False
    with read_only(path) as conn:
        row = conn.execute(
            "SELECT ever_existed FROM board_name_marker WHERE board_name = ?",
            (slug,),
        ).fetchone()
    return bool(row is not None and row["ever_existed"])


def archive_records(slug: str) -> int:
    path = kb.archive_db_path()
    if not path.exists():
        return 0
    with read_only(path) as conn:
        return int(
            conn.execute(
                "SELECT COUNT(*) FROM board_removal_archive WHERE board_name = ?",
                (slug,),
            ).fetchone()[0]
        )


def intent_rows(slug: str) -> int:
    path = kb.archive_db_path()
    if not path.exists():
        return 0
    with read_only(path) as conn:
        return int(
            conn.execute(
                "SELECT COUNT(*) FROM board_backfill_intent WHERE board_name = ?",
                (slug,),
            ).fetchone()[0]
        )


def intent_row(slug: str):
    """The board's backfill-intent journal row as persisted, or None."""
    path = kb.archive_db_path()
    if not path.exists():
        return None
    with read_only(path) as conn:
        try:
            row = conn.execute(
                "SELECT intent_id, epoch, phase, gate_b_preexisting "
                "FROM board_backfill_intent WHERE board_name = ?",
                (slug,),
            ).fetchone()
        except sqlite3.Error:
            return None
    return None if row is None else dict(row)


def row_count(db_path: Path, table: str) -> int:
    with read_only(db_path) as conn:
        return int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


def task_row(db_path: Path, task_id: str):
    with read_only(db_path) as conn:
        row = conn.execute(
            "SELECT status, claim_lock, claim_expires FROM tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
    return None if row is None else dict(row)


# ---------------------------------------------------------------------------
# Real production drivers
# ---------------------------------------------------------------------------

def cli(command: str) -> str:
    """Run a real CLI invocation and return its captured output.

    ``run_slash`` is the shipped entry point the interactive CLI and the
    gateway both dispatch through: it builds the real argparse tree and
    calls :func:`hermes_cli.kanban.kanban_command`.
    """
    from hermes_cli.kanban import run_slash

    return run_slash(command)


def ready_task(conn, title: str = "work", assignee: str = "worker") -> str:
    """Create a task through the real API and park it in ``ready``."""
    task_id = kb.create_task(conn, title=title, assignee=assignee)
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (task_id,))
    return task_id


def create_fenced_board(slug: str, **kwargs) -> None:
    """Create a board and arm its fence via the real recorded migration.

    ``create_board`` no longer arms a fence on its own: a freshly created
    board carries no register entry and no in-board gate until an
    operator runs the backfill. Tests that need a FENCED board to
    exercise Gate A/Gate B behaviour go through this, the same recorded
    path (``hermes kanban boards backfill-fence``) an operator would use.
    """
    kb.create_board(slug, **kwargs)
    result = kb.backfill_register_entry(slug)
    assert result.success, result.message


def permanent_confirmation(slug: str) -> "kb.PermanentRemovalConfirmation":
    """The operator confirmation §6.1 requires for a permanent removal.

    Goes through the shipped surfaces, in the order a real operator does:
    :func:`kanban_db.permanent_removal_disclosure` builds the statement
    and the exact answer bound to it, and
    :func:`kanban_db.confirm_permanent_removal` mints the confirmation
    from that answer. Tests never hand-build a confirmation, or they
    would stop proving that the real statement is what gets confirmed —
    and they never mint one without an answer, or they would stop proving
    that an answer is required at all.
    """
    disclosure = kb.permanent_removal_disclosure(slug)
    checked = kb.confirm_permanent_removal(
        slug,
        response=disclosure.required_response,
        confirmed_by="operator",
        disclosure=disclosure,
    )
    assert checked.confirmed, checked.message
    return checked.confirmation


def start_removal(slug: str, *, mode: str = "reversible", removal_id=None):
    """``record_removal_intent`` with whatever that mode requires (§6.1).

    Reversible needs nothing; permanent needs the operator confirmation.
    Returns the real :class:`kanban_db.RemovalIntentResult`.
    """
    kwargs = {}
    if kb.RemovalMode(mode) == kb.RemovalMode.PERMANENT:
        kwargs["permanent_confirmation"] = permanent_confirmation(slug)
    return kb.record_removal_intent(
        slug, mode=mode, removal_id=removal_id, **kwargs
    )


def apply_mode_specific_content(slug: str, removal_id: str):
    """Do §6.6's MODE-SPECIFIC work for real, then record it via the seam.

    §7.2 (permanent: destroy the storage, deregister the registrations)
    now ships as :func:`kanban_db.apply_permanent_mode_content`; §7.3
    (reversible: complete and verify the retained copy, remove the live
    one) is still later work. Done refuses while the mode's content is
    outstanding, so a test that is about Done has to make that content
    actually TRUE. Tests covering the PERMANENT path should drive the
    shipped §7.2 driver instead of this helper; this remains for the
    reversible path and for tests that need the content true without
    exercising the driver itself, which is all it does: it removes the board's storage (permanent), or copies it
    somewhere outside the board and then removes the live one
    (reversible), and hands the result to the shipped
    :func:`kanban_db.record_applied_mode_content`, which verifies the
    claim against the filesystem itself.

    Deliberately NOT a helper that writes the marker: a helper that could
    record "applied" without the work having happened would be exactly
    the caller-trusted flag the seam refuses, and every Done test built
    on it would be asserting the defect again.
    """
    record = kb.get_removal_phase_record(slug)
    assert record is not None, f"{slug} has no removal phase record"
    board = kb.board_dir(slug)
    retained = None
    if record.mode is kb.RemovalMode.REVERSIBLE:
        retained = kb.kanban_home() / "retained" / f"{slug}-{removal_id}"
        retained.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(board, retained)
    if board.exists():
        shutil.rmtree(board)
    db_path = kb.kanban_db_path(board=slug)
    if db_path.exists():  # the default board's store lives outside board_dir
        db_path.unlink()
    result = kb.record_applied_mode_content(
        slug,
        removal_id=removal_id,
        evidence=kb.AppliedModeContentEvidence(
            performed_by="test standing in for the §7.2/§7.3 driver",
            retained_path=None if retained is None else str(retained),
            detail="the mode-specific content was performed on disk",
        ),
    )
    assert result.success, result.message
    return result


# ---------------------------------------------------------------------------
# Real git repositories, real linked work areas, real commit receipts
# ---------------------------------------------------------------------------

_GIT_ENV = {
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_SYSTEM": os.devnull,
    "GIT_TERMINAL_PROMPT": "0",
}


def git(repo: Path, *argv: str, author: "tuple[str, str] | None" = None) -> str:
    """Run a REAL git command in *repo*, isolated from the host's config.

    ``author`` sets ``user.name`` / ``user.email`` for THIS invocation
    only, which is how a genuinely foreign-identity commit is made: the
    commit really carries a different author and committer, not a label
    a test asserted about itself.
    """
    identity: list = []
    if author is not None:
        name, email = author
        identity = [
            "-c", f"user.name={name}", "-c", f"user.email={email}",
        ]
    proc = subprocess.run(
        ["git", "-C", str(repo), *identity, *argv],
        capture_output=True, text=True, timeout=60,
        env={**os.environ, **_GIT_ENV},
    )
    assert proc.returncode == 0, (
        f"git {' '.join(argv)} failed in {repo}: {proc.stderr or proc.stdout}"
    )
    return proc.stdout.strip()


DEFAULT_GIT_IDENTITY = ("Board Owner", "owner@example.invalid")
FOREIGN_GIT_IDENTITY = ("Someone Else", "someone-else@example.invalid")


def make_git_repo(path: Path, *, author=DEFAULT_GIT_IDENTITY) -> str:
    """Initialise a real repository with one real commit. Returns its head.

    Idempotent: a repository that already exists keeps its history and
    its head is returned, so two boards can share one container.
    """
    path.mkdir(parents=True, exist_ok=True)
    if (path / ".git").exists():
        return git(path, "rev-parse", "main")
    git(path, "init", "--initial-branch=main")
    (path / "README.md").write_text("seed\n", encoding="utf-8")
    git(path, "add", "README.md")
    git(path, "commit", "-m", "seed", author=author)
    return git(path, "rev-parse", "HEAD")


def add_linked_work_area(
    repo: Path, name: str, *, branch: str, content: str = "work\n",
    author=DEFAULT_GIT_IDENTITY,
) -> "tuple[Path, str]":
    """Add a REAL linked git work area at ``<repo>/.worktrees/<name>``.

    That is the exact shape ``create_task`` gives a project-linked
    worktree task, so the container's own registration entry
    (``<repo>/.git/worktrees/<name>/gitdir``) is written by git itself —
    which is what the ownership check reads.

    Returns ``(work_area, head_commit)``; the head commit is a real
    commit made by *author*, so a foreign identity is genuinely foreign.
    """
    work_area = repo / ".worktrees" / name
    work_area.parent.mkdir(parents=True, exist_ok=True)
    git(repo, "worktree", "add", "-b", branch, str(work_area))
    (work_area / f"{name}.txt").write_text(content, encoding="utf-8")
    git(work_area, "add", f"{name}.txt")
    git(work_area, "commit", "-m", f"work for {name}", author=author)
    return work_area, git(work_area, "rev-parse", "HEAD")


def commit_series(
    work_area: Path, count: int, *, prefix: str = "step",
    author=DEFAULT_GIT_IDENTITY,
) -> list:
    """Make *count* REAL, sequential commits in *work_area*.

    Returns the commit identities in the order git created them, so a test
    can name an INTERMEDIATE commit — the one an advance that recorded only
    its head does not account for.
    """
    heads: list = []
    for index in range(count):
        name = f"{prefix}-{index}.txt"
        (work_area / name).write_text(f"{prefix} {index}\n", encoding="utf-8")
        git(work_area, "add", name)
        git(work_area, "commit", "-m", f"{prefix} {index}", author=author)
        heads.append(git(work_area, "rev-parse", "HEAD"))
    return heads


def board_with_multi_commit_advance(
    slug: str,
    repo: Path,
    *,
    extra_commits: int = 2,
    absorbed_heads: "list | None" = None,
    author=DEFAULT_GIT_IDENTITY,
    title: str = "several commits in one advance",
) -> dict:
    """A fenced board whose ONE recorded advance produced SEVERAL commits.

    Real repository, real linked work area, real sequential commits, and
    ONE run receipt naming only the advance's head — which is exactly the
    shape a receipt that listed only the final head would gloss over. The
    intermediate commits are returned so a test can assert what is and is
    not accounted for.
    """
    base = make_git_repo(repo, author=author)
    create_fenced_board(slug)
    conn = kb.connect(board=slug)
    task_id = kb.create_task(conn, title=title, assignee="worker")
    branch = f"hermes/{task_id}"
    work_area, first_head = add_linked_work_area(
        repo, task_id, branch=branch, author=author,
    )
    later = commit_series(work_area, extra_commits, author=author)
    head = later[-1] if later else first_head
    run_id = record_run_advance(
        conn, task_id, branch=branch, base_commit=base, head_commit=head,
        absorbed_heads=absorbed_heads,
    )
    record_git_receipt(
        conn, task_id, workspace_path=work_area, branch_name=branch,
        base_commit=base, head_commit=head,
    )
    conn.close()
    return {
        "slug": slug,
        "task": task_id,
        "repo": repo,
        "branch": branch,
        "base": base,
        "head": head,
        # Every commit the advance really created, oldest first. All but
        # the last are INTERMEDIATE: real commits no recorded receipt names.
        "created": [first_head, *later],
        "intermediate": [first_head, *later][:-1],
        "work_area": work_area,
        "registration": repo / ".git" / "worktrees" / task_id,
        "run": run_id,
    }


def record_git_receipt(
    conn,
    task_id: str,
    *,
    workspace_path: Path,
    branch_name: str,
    base_commit: str,
    head_commit: str,
    project_id: str = "test-project",
) -> None:
    """Persist the git receipt columns a completed worktree task carries.

    Arranges the DURABLE STATE a completion leaves behind — the columns
    ``complete_task`` writes — so a test can then exercise the removal
    behaviour that reads them. Same shape as :func:`ready_task`'s status
    write: state arrangement through the real write transaction, never a
    substitute for the behaviour under test.
    """
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET workspace_kind = 'worktree', workspace_path = ?, "
            "branch_name = ?, base_commit = ?, head_commit = ?, "
            "project_id = ?, status = 'done' WHERE id = ?",
            (
                str(workspace_path), branch_name, base_commit, head_commit,
                project_id, task_id,
            ),
        )


def record_run_advance(
    conn,
    task_id: str,
    *,
    branch: str,
    base_commit: str,
    head_commit: str,
    absorbed_heads: "list | None" = None,
) -> int:
    """Persist ONE run's execution receipt — one advance of this board's work.

    The same ``task_runs.metadata['execution_receipt']`` shape
    ``complete_task`` persists, including ``parent_heads``: the heads
    this advance ABSORBED from another subject's work.
    """
    import json as _json

    receipt = {
        "kind": "scoped_worktree_v1",
        "base_commit": base_commit,
        "head_commit": head_commit,
        "branch": branch,
        "owned_paths": ["."],
        "changed_paths": [],
        "parent_heads": [
            {"task_id": head["task"], "head_commit": head["head_commit"]}
            for head in (absorbed_heads or [])
        ],
    }
    now = int(time.time())
    with kb.write_txn(conn):
        cur = conn.execute(
            "INSERT INTO task_runs (task_id, profile, status, started_at, "
            "ended_at, outcome, metadata) VALUES (?, 'worker', 'done', ?, ?, "
            "'completed', ?)",
            (task_id, now, now, _json.dumps({"execution_receipt": receipt})),
        )
    return int(cur.lastrowid)


def begin_removal(slug: str) -> "kb.RegisterEntry":
    """Phase 1 of a removal: the register intent, through the real primitive."""
    entry = kb.get_register_entry(slug)
    assert entry is not None, f"{slug} has no register entry to remove"
    new_epoch = entry.epoch + 1
    updated = kb.RegisterEntry(
        board_name=slug,
        lifecycle=kb.BoardLifecycle.REMOVING,
        epoch=new_epoch,
        epoch_before=entry.epoch,
        gate_move=kb.GateMove.PENDING,
        epoch_lineage=(entry.epoch_lineage or []) + [new_epoch],
        created_at=entry.created_at,
        updated_at=int(time.time()),
    )
    kb.transition_register_entry(updated)
    return updated


def close_fence(slug: str) -> None:
    """Drive a board to the fence-closing point through the real primitives."""
    begin_removal(slug)
    result = kb.commit_fence_closing_point(slug)
    assert result.success and result.transitioned, result.message


def write_register_row_behind_the_lock(entry: "kb.RegisterEntry") -> None:
    """Change Gate A WITHOUT taking the per-board register lock.

    Stands in for the writer the revalidation exists to catch: a crashed
    half-write, an operator repair, or any transition that did not go
    through :func:`kanban_db.transition_register_entry`.
    """
    with kb.register_connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
            kb._write_register_entry(conn, entry)
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise


def damage_phase_column_behind_the_primitive(slug: str, column: str, value) -> None:
    """Overwrite ONE column of the durable phase record, past the primitive.

    The phase record's peer of :func:`write_register_row_behind_the_lock`.
    It stands in for the writer no code path is allowed to be — a crashed
    half-write, an operator repair, a row restored from a backup taken
    mid-run, bit-rot — which is precisely the state each driver's
    revalidation of its own durable completion fact exists to catch.

    ``phase`` itself is refused here: the compare-and-set primitive is the
    only writer of that column, so no test may use this to manufacture a
    phase the primitive would have refused. Only the durable FACT a
    phase's meaning consists of can be damaged.
    """
    assert column != "phase", "the compare-and-set is the only writer of phase"
    assert column in kb._REMOVAL_PHASE_RECORD_COLUMNS, f"unknown column {column!r}"
    with kb.register_connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
            cur = conn.execute(
                f"UPDATE board_removal_phase SET {column} = ? WHERE board_name = ?",
                (value, slug),
            )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
    assert cur.rowcount == 1, f"no removal phase record for {slug!r}"


def make_legacy_board(tmp_path: Path, slug: str) -> Path:
    """A board that genuinely predates the fence: schema, no Gate B, no entry.

    Built by creating a store at a path with NO board identity — the
    arbitrary-path compatibility connector, which is outside the fence —
    and then moving it into place as a board directory, exactly the shape
    an install upgraded from a pre-fence release has on disk.
    """
    staging = stage_legacy_db(tmp_path, slug)
    target = kb.kanban_db_path(board=slug)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(staging), str(target))
    kb._INITIALIZED_PATHS.clear()
    kb.write_board_metadata(slug)
    assert not has_gate_table(target)
    assert kb.get_register_entry(slug) is None
    return target


LEGACY_SENTINEL_TITLE = "work from before the fence"


def stage_legacy_db(tmp_path: Path, slug: str) -> Path:
    """A valid pre-fence store, built but NOT yet moved into a board.

    Same construction as :func:`make_legacy_board`, stopped one step
    earlier so a test can decide WHEN the file appears at the board path.
    Carries one sentinel task, so "the legacy data survived" is something
    the assertions can actually read back rather than infer from a size.
    """
    staging = Path(tmp_path) / f"legacy-{slug}.db"
    with contextlib.closing(kb.connect(db_path=staging)) as conn:
        kb.create_task(conn, title=LEGACY_SENTINEL_TITLE, assignee="worker")
    kb._INITIALIZED_PATHS.discard(str(staging.resolve()))
    return staging


def make_orphan_gate_board(tmp_path: Path, slug: str, *, mirror: int) -> Path:
    """A legacy board that ALREADY carries a Gate B nobody vouches for.

    The shape a crashed backfill leaves behind: the in-board gate table is
    installed (by the shipped primitive that installs it) and open, while
    the register has no authority for the name at all. ``mirror`` is set
    to a value no fresh arming would produce, so a later assertion can
    tell "the prior gate was put back" apart from "our epoch was left in
    place".
    """
    target = make_legacy_board(tmp_path, slug)
    conn = kb._sqlite_connect_no_create(target)
    try:
        conn.execute("BEGIN IMMEDIATE")
        kb._ensure_in_board_fence_schema(conn)
        conn.execute(
            "UPDATE board_fence_state SET epoch_mirror = ? WHERE id = 1",
            (int(mirror),),
        )
        conn.execute("COMMIT")
    finally:
        conn.close()
    assert gate_row(target) == ("open", int(mirror))
    return target


# ---------------------------------------------------------------------------
# Real held locks (a separate OS process, so the lock is real)
# ---------------------------------------------------------------------------

_HOLDER_SOURCE = """
import sqlite3, sys, time
path, mode = sys.argv[1], sys.argv[2]
con = sqlite3.connect(path, isolation_level=None, timeout=1)
con.execute("PRAGMA busy_timeout=1000")
if mode == "exclusive-file":
    con.execute("PRAGMA locking_mode=EXCLUSIVE")
    con.execute("BEGIN IMMEDIATE")
    con.execute("CREATE TABLE IF NOT EXISTS _lock_probe (id INTEGER)")
elif mode == "reserved":
    con.execute("BEGIN IMMEDIATE")
else:
    con.execute("BEGIN EXCLUSIVE")
print("locked", flush=True)
time.sleep(120)
"""


@contextlib.contextmanager
def hold_write_lock(db_path: Path):
    """Hold a real WRITER lock on *db_path* from another process.

    ``BEGIN IMMEDIATE`` takes SQLite's RESERVED lock: other processes may
    still READ the store, but any other writer blocks. That asymmetry is
    what lets a test park a real cross-store protocol at its next WRITE
    while still watching the same store's persisted state go by.

    Yields a ``release()`` callable so the parked writer can be let go at
    a chosen point; the lock is released on exit either way.
    """
    proc = subprocess.Popen(
        [sys.executable, "-c", _HOLDER_SOURCE, str(db_path), "reserved"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    released = False

    def release() -> None:
        nonlocal released
        if released:
            return
        released = True
        proc.terminate()
        with contextlib.suppress(Exception):
            proc.wait(timeout=10)

    try:
        line = proc.stdout.readline().strip()
        assert line == "locked", f"write-lock holder failed to start: {line!r}"
        yield release
    finally:
        release()


@contextlib.contextmanager
def hold_exclusive_lock(db_path: Path, *, whole_file: bool = False):
    """Hold a real SQLite write lock on *db_path* from another process.

    ``whole_file=True`` takes ``locking_mode=EXCLUSIVE`` so even READS
    from other processes block — which is what a contended Gate A looks
    like to a mutation trying to consult it.
    """
    mode = "exclusive-file" if whole_file else "exclusive-txn"
    proc = subprocess.Popen(
        [sys.executable, "-c", _HOLDER_SOURCE, str(db_path), mode],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    try:
        line = proc.stdout.readline().strip()
        assert line == "locked", f"lock holder failed to start: {line!r}"
        yield proc
    finally:
        proc.terminate()
        with contextlib.suppress(Exception):
            proc.wait(timeout=10)
