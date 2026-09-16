"""A removed board is never brought back by looking at it (GA-2, GA-4).

``connect()`` used to ``mkdir`` the board directory and then open with a
creating connect, so any poll loop, watcher, stale handle or CLI auto-init
that merely LOOKED at a board a removal had just taken away re-created an
empty store at that exact path — a board that was removed reopened, empty
and unfenced. Opening is now separate from creating: only the explicit
creation path (``init_db``, and ``create_board`` through it) may bring a
store into existence, and it refuses to do so for a board the register
says is being or has been removed.

Everything here drives the shipped APIs — ``create_board`` + the recorded
backfill, ``remove_board_fenced``, ``connect``, ``claim_task`` — and reads
the outcome off the filesystem or through a plain read-only connection,
never through the module under test. Every test carries a CONTROL board
that is not removed, because "nothing can open" would pass every assertion
about the removed one.
"""

from __future__ import annotations

import contextlib
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_WORKTREE = Path(__file__).resolve().parents[2]
if str(_WORKTREE) not in sys.path:
    sys.path.insert(0, str(_WORKTREE))

from hermes_cli import kanban_db as kb
from hermes_cli.sqlite_util import (
    InitLockDirectoryAbsent,
    cross_process_init_lock,
)
from tests.hermes_cli._kanban_fence_support import (
    begin_removal,
    close_fence,
    create_fenced_board,
    permanent_confirmation,
    ready_task,
    register_row,
    start_removal,
    task_row,
)

_CHILD = Path(__file__).parent / "_kanban_removal_crash_child.py"


def _remove_in_another_process(home: Path, slug: str) -> None:
    """Remove *slug* through the shipped CLI in a REAL separate process.

    This process's ``_INITIALIZED_PATHS`` cache therefore never learns the
    board went away — which is exactly the state a long-lived gateway or
    dashboard is in when an operator removes a board from a terminal.
    """
    proc = subprocess.run(
        [sys.executable, str(_CHILD), str(home), slug, "never", "reversible"],
        capture_output=True, text=True, timeout=180,
        cwd=str(_WORKTREE),
        env={
            "PATH": "/usr/bin:/bin:/usr/local/bin",
            "PYTHONPATH": str(_WORKTREE),
            "PYTHONHASHSEED": "0",
            "TZ": "UTC",
            "LANG": "C.UTF-8",
            "HOME": str(home),
        },
    )
    assert "CHILD-EXIT 0" in proc.stdout, (
        f"child removal failed: {proc.stdout}\n{proc.stderr}"
    )


def _control_board_still_works(slug: str) -> None:
    """The control board opens, claims and writes — throughout and after."""
    conn = kb.connect(board=slug)
    try:
        task_id = ready_task(conn, title="control work")
        assert kb.claim_task(conn, task_id) is not None
    finally:
        conn.close()
    assert task_row(kb.kanban_db_path(board=slug), task_id)["status"] == "running"


def _residue(slug: str) -> list:
    """Everything discoverable at the board's path, read off the filesystem."""
    directory = kb.board_dir(slug)
    if not directory.exists():
        return []
    return sorted(p.name for p in directory.iterdir())


def _assert_left_nothing_behind(slug: str, *, what: str) -> None:
    """No board directory, no store, no init lock, no metadata.

    A refusal that still creates SOMETHING at the removed board's path has
    not refused: the directory alone makes the name discoverable again to
    ``list_boards``, and an ``init.lock`` or a ``board.json`` is a file a
    removal already accounted for and destroyed.
    """
    directory = kb.board_dir(slug)
    db_path = kb.kanban_db_path(board=slug)
    init_lock = db_path.with_name(db_path.name + ".init.lock")
    assert not db_path.exists(), f"{what} re-created the store: {db_path}"
    assert not init_lock.exists(), f"{what} re-created the init lock: {init_lock}"
    assert not kb.board_metadata_path(slug).exists(), (
        f"{what} re-created board.json"
    )
    assert not directory.exists(), (
        f"{what} re-created the board directory, holding {_residue(slug)}"
    )


@contextlib.contextmanager
def _removal_lands_at_the_init_lock(home: Path, slug: str):
    """Run a REAL removal in the window this boundary is about.

    The caller has already observed that the path exists; the removal (in
    another process, through the shipped CLI) completes immediately before
    the cross-process init lock is entered. Everything the opening process
    decided — "the store is there", "``create=True`` says I may make one" —
    is stale from this point on, and the init lock is the next thing that
    would touch the filesystem at that path.

    Yields a one-element list recording whether the interleaving actually
    happened, so a test can never pass by never reaching the window.
    """
    path = kb.kanban_db_path(board=slug)
    original = kb._cross_process_init_lock
    reached: list = []
    kb._INITIALIZED_PATHS.clear()

    @contextlib.contextmanager
    def remove_then_lock(target):
        if target == path and not reached:
            reached.append(True)
            _remove_in_another_process(home, slug)
            assert not path.parent.exists(), "the removal did not take the board"
        with original(target):
            yield

    kb._cross_process_init_lock = remove_then_lock
    try:
        yield reached
    finally:
        kb._cross_process_init_lock = original


# ---------------------------------------------------------------------------
# GA-4 — an open never creates
# ---------------------------------------------------------------------------

def test_a_cold_open_of_a_removed_board_creates_nothing(fence_home):
    """First open of the path in this process: no directory, no store."""
    create_fenced_board("doomed")
    create_fenced_board("control")
    conn = kb.connect(board="doomed")
    ready_task(conn)
    conn.close()

    db_path = kb.kanban_db_path(board="doomed")
    board_dir = kb.board_dir("doomed")
    assert kb.remove_board_fenced("doomed", mode="reversible").success
    assert not board_dir.exists()

    # A cold process knows nothing about this path.
    kb._INITIALIZED_PATHS.clear()
    with pytest.raises(kb.BoardFenceClosedError) as excinfo:
        kb.connect(board="doomed")

    assert excinfo.value.refusal.rule is kb.FenceRefusalRule.GA_4
    assert not db_path.exists(), "the open re-created the store"
    assert not board_dir.exists(), "the open re-created the board directory"
    _control_board_still_works("control")


def test_the_cached_fast_path_does_not_recreate_a_removed_board(fence_home):
    """The in-process cache says "initialized"; the board is gone anyway.

    The removal runs in a REAL separate process, so this process's cache
    entry survives it — the fast path is genuinely taken, which is the
    path a long-lived gateway or dashboard would have resurrected the
    board from.
    """
    create_fenced_board("cached")
    create_fenced_board("control")
    conn = kb.connect(board="cached")
    ready_task(conn)
    conn.close()

    db_path = kb.kanban_db_path(board="cached")
    board_dir = kb.board_dir("cached")
    assert str(db_path.resolve()) in kb._INITIALIZED_PATHS

    _remove_in_another_process(fence_home, "cached")
    assert not board_dir.exists()
    assert str(db_path.resolve()) in kb._INITIALIZED_PATHS, (
        "this process must still believe the path is initialized, or the "
        "cached fast path is not the one under test"
    )

    with pytest.raises(kb.BoardFenceClosedError) as excinfo:
        kb.connect(board="cached")

    assert excinfo.value.refusal.rule is kb.FenceRefusalRule.GA_4
    assert not db_path.exists(), "the cached open re-created the store"
    assert not board_dir.exists(), "the cached open re-created the directory"
    _control_board_still_works("control")


def test_auto_init_refuses_to_recreate_a_board_under_removal(fence_home):
    """GA-2: even the explicit creation path refuses a removed board.

    ``hermes kanban <anything>`` auto-inits the current board on every
    invocation, so a removal that has already destroyed the storage would
    otherwise be undone by the next command an operator typed.
    """
    create_fenced_board("auto-init")
    create_fenced_board("control")
    assert kb.remove_board_fenced("auto-init", mode="reversible").success

    with pytest.raises(kb.BoardFenceClosedError) as excinfo:
        kb.init_db(board="auto-init")

    assert excinfo.value.refusal.rule is kb.FenceRefusalRule.GA_2
    assert not kb.board_dir("auto-init").exists()
    assert not kb.kanban_db_path(board="auto-init").exists()
    # The control board's own auto-init is untouched.
    assert kb.init_db(board="control").exists()
    _control_board_still_works("control")


def test_creating_a_never_removed_board_still_works(fence_home):
    """The creation path is only closed to boards the register has removed.

    Without this, "creation is refused" would be indistinguishable from
    "creation is broken".
    """
    meta = kb.create_board("brand-new")
    assert meta["slug"] == "brand-new"
    assert kb.kanban_db_path(board="brand-new").exists()
    _control_board_still_works("brand-new")


# ---------------------------------------------------------------------------
# The admission/creation boundary — a removal that lands mid-flight
#
# Every check above decides from an observation the caller took BEFORE the
# window it acts in. These drive the window itself: a real removal, in a real
# second process, completing after the caller has already seen the path and
# immediately before the cross-process init lock — the next thing that would
# touch the filesystem at that path. A refusal that still leaves a directory,
# a store, an ``init.lock`` or a ``board.json`` behind has resurrected the
# board's discoverable identity even though it raised.
# ---------------------------------------------------------------------------

def test_an_open_that_loses_the_race_does_not_let_the_init_lock_recreate_it(
    fence_home,
):
    """The init lock is a sibling of the store: it must not be what revives it.

    ``connect(create=False)`` already refused this board — but acquiring
    ``kanban.db.init.lock`` ``mkdir``-ed the removed board's directory and
    put a lock file in it first, so the refusal arrived after the directory
    was already back.
    """
    create_fenced_board("lock-race")
    create_fenced_board("control")
    conn = kb.connect(board="lock-race")
    ready_task(conn)
    conn.close()

    with _removal_lands_at_the_init_lock(fence_home, "lock-race") as reached:
        with pytest.raises(kb.BoardFenceClosedError) as excinfo:
            kb.connect(board="lock-race").close()

    assert reached, "the removal never landed in the window under test"
    assert excinfo.value.refusal.rule is kb.FenceRefusalRule.GA_4
    _assert_left_nothing_behind("lock-race", what="the refused open")
    _control_board_still_works("control")


def test_auto_init_that_loses_the_race_does_not_resurrect_the_board(fence_home):
    """``create=True`` decided before the removal is not authority after it.

    ``hermes kanban <anything>`` auto-inits the current board, so this is
    the ordinary CLI path, not just an explicit ``init``. Admission used to
    be decided from the up-front ``path.exists()`` snapshot and never
    re-read, so a removal landing after it was overtaken by a ``create=True``
    that predated it — and ``init_db`` SUCCEEDED, rebuilding both the store
    and its lock.
    """
    create_fenced_board("auto-race")
    create_fenced_board("control")
    conn = kb.connect(board="auto-race")
    ready_task(conn)
    conn.close()

    with _removal_lands_at_the_init_lock(fence_home, "auto-race") as reached:
        with pytest.raises(kb.BoardFenceClosedError) as excinfo:
            kb.init_db(board="auto-race")

    assert reached, "the removal never landed in the window under test"
    assert excinfo.value.refusal.rule is kb.FenceRefusalRule.GA_4
    _assert_left_nothing_behind("auto-race", what="the refused auto-init")
    _control_board_still_works("control")


def test_create_board_of_a_removed_name_writes_no_metadata(fence_home):
    """The removed-lifecycle check comes BEFORE ``board.json`` is written.

    ``create_board`` wrote the metadata first and let ``init_db`` refuse
    second, so the call raised with a directory and a ``board.json`` already
    published for a name the register had removed — a ghost ``list_boards``
    still discovers.
    """
    create_fenced_board("ghosted")
    create_fenced_board("control")
    _remove_in_another_process(fence_home, "ghosted")
    assert not kb.board_dir("ghosted").exists()

    with pytest.raises(kb.BoardFenceClosedError) as excinfo:
        kb.create_board("ghosted")

    assert excinfo.value.refusal.rule is kb.FenceRefusalRule.GA_2
    _assert_left_nothing_behind("ghosted", what="the refused create_board")
    assert all(entry["slug"] != "ghosted" for entry in kb.list_boards())
    _control_board_still_works("control")


def test_a_creation_refused_mid_flight_takes_back_what_it_created(fence_home):
    """Admitted at the door, refused at the moment of creation: no residue.

    The storage is gone while the register still says ``live`` — the state a
    crash between "removal destroyed the store" and "removal recorded it"
    leaves — so ``create_board`` is genuinely admitted when it starts and
    really does make the directory and ``board.json``. The register then
    moves to ``removing`` (through the real Gate A primitive) inside the
    window, which is where admission is re-read. The refusal has to undo
    everything the refused creation put on disk, including the init-lock
    file the lock itself materialised.
    """
    create_fenced_board("mid-flight")
    create_fenced_board("control")
    directory = kb.board_dir("mid-flight")
    db_path = kb.kanban_db_path(board="mid-flight")
    shutil.rmtree(directory)
    kb._INITIALIZED_PATHS.clear()
    assert kb.get_register_entry("mid-flight").lifecycle is kb.BoardLifecycle.LIVE

    original = kb._cross_process_init_lock
    moved: list = []

    @contextlib.contextmanager
    def close_gate_a_then_lock(target):
        if target == db_path and not moved:
            moved.append(True)
            begin_removal("mid-flight")
            assert directory.exists(), (
                "the creation must already have made the directory, or this "
                "is not the window under test"
            )
        with original(target):
            yield

    kb._cross_process_init_lock = close_gate_a_then_lock
    try:
        with pytest.raises(kb.BoardFenceClosedError) as excinfo:
            kb.create_board("mid-flight")
    finally:
        kb._cross_process_init_lock = original

    assert moved, "Gate A never moved inside the window under test"
    assert excinfo.value.refusal.rule is kb.FenceRefusalRule.GA_2
    _assert_left_nothing_behind("mid-flight", what="the refused creation")
    _control_board_still_works("control")


def test_a_cold_open_of_an_existing_store_mid_removal_is_still_admitted(
    fence_home,
):
    """The boundary closed CREATION, not opening — GA-1 is not over-corrected.

    A removal that has begun but has not yet taken the storage away must
    still let already-accepted work reach its store; the write chokepoint,
    not the open, is what decides whether it may still mutate. Without this,
    "creation is refused" and "the board is unreachable" would be
    indistinguishable.
    """
    create_fenced_board("still-there")
    create_fenced_board("control")
    conn = kb.connect(board="still-there")
    task_id = ready_task(conn)
    conn.close()

    begin_removal("still-there")
    assert kb.board_dir("still-there").exists()
    kb._INITIALIZED_PATHS.clear()

    conn = kb.connect(board="still-there")
    try:
        row = conn.execute(
            "SELECT status FROM tasks WHERE id = ?", (task_id,)
        ).fetchone()
    finally:
        conn.close()
    assert row["status"] == "ready", "the admitted open could not read its store"
    _control_board_still_works("control")


# ---------------------------------------------------------------------------
# Admission authority — the board admitted is the one at the TARGET PATH
#
# Admission read the ``board`` ARGUMENT first and only fell back to the path,
# so a caller naming a LIVE board while targeting a REMOVED board's file was
# admitted against the live name and rebuilt the removed board's store.
# ``HERMES_KANBAN_DB`` is the same defect without an explicit argument: it
# outranks ``board`` when the path is resolved, so the file that would be
# created and the name that was checked came from two different boards.
#
# Both cases drive ``init_db`` — the one route that asks for ``create=True``
# — after a REAL removal in a REAL second process, and read the outcome off
# the filesystem. The last test is the other half of the contract: a pinned
# database the register has never heard of must still bootstrap, or
# "the path decides" would mean "nothing outside the boards tree may exist".
# ---------------------------------------------------------------------------


def test_a_live_board_name_cannot_create_a_removed_boards_store(fence_home):
    """An explicit ``db_path`` at a removed board, named as a live one.

    The name is genuinely live and genuinely admissible — for its OWN
    store. What decides admission here is the file the creation would
    actually bring into existence, which belongs to a board the register
    has removed.
    """
    create_fenced_board("named-elsewhere")
    create_fenced_board("control")
    conn = kb.connect(board="named-elsewhere")
    ready_task(conn)
    conn.close()
    removed_db = kb.kanban_db_path(board="named-elsewhere")

    _remove_in_another_process(fence_home, "named-elsewhere")
    _assert_left_nothing_behind("named-elsewhere", what="the removal")

    with pytest.raises(kb.BoardFenceClosedError) as excinfo:
        kb.init_db(db_path=removed_db, board="control")

    assert excinfo.value.refusal.rule is kb.FenceRefusalRule.GA_2
    assert excinfo.value.refusal.board == "named-elsewhere", (
        "the refusal is about the board at the path, not the one named"
    )
    _assert_left_nothing_behind("named-elsewhere", what="the misnamed init_db")
    _control_board_still_works("control")


def test_a_live_board_name_cannot_create_a_pinned_removed_store(
    fence_home, monkeypatch,
):
    """The same authority, reached through ``HERMES_KANBAN_DB``.

    No ``db_path`` argument at all: the pin is what ``kanban_db_path``
    resolves to, above the ``board`` argument. The dispatcher injects this
    variable into worker environments, so a worker inheriting a pin for a
    board that was removed meanwhile is the production shape of this.
    """
    create_fenced_board("pinned-away")
    create_fenced_board("control")
    conn = kb.connect(board="pinned-away")
    ready_task(conn)
    conn.close()
    removed_db = kb.kanban_db_path(board="pinned-away")

    _remove_in_another_process(fence_home, "pinned-away")
    _assert_left_nothing_behind("pinned-away", what="the removal")

    with monkeypatch.context() as pinned:
        pinned.setenv("HERMES_KANBAN_DB", str(removed_db))
        with pytest.raises(kb.BoardFenceClosedError) as excinfo:
            kb.init_db(board="control")

    assert excinfo.value.refusal.rule is kb.FenceRefusalRule.GA_2
    assert excinfo.value.refusal.board == "pinned-away", (
        "the refusal is about the board at the pinned path, not the one named"
    )
    _assert_left_nothing_behind("pinned-away", what="the pinned init_db")
    _control_board_still_works("control")


def test_a_foreign_pinned_database_is_still_created_and_usable(
    fence_home, tmp_path, monkeypatch,
):
    """The fallback the path-first authority has to keep.

    ``HERMES_KANBAN_DB`` legitimately pins legacy and out-of-tree files.
    Such a path carries no board identity of its own, so admission falls
    back to the ``board`` argument — and with no register entry behind it
    there is nothing to refuse. Without this, every check above would also
    pass if the boundary had simply stopped creating anything.
    """
    create_fenced_board("control")
    legacy = tmp_path / "legacy-elsewhere" / "kanban-legacy.db"
    assert not legacy.parent.exists()
    assert kb.board_slug_for_db_path(legacy) is None, (
        "the register must never have heard of this path, or the fallback "
        "is not what is under test"
    )

    with monkeypatch.context() as pinned:
        pinned.setenv("HERMES_KANBAN_DB", str(legacy))
        assert kb.init_db() == legacy
        assert legacy.exists(), "the legacy pin was not bootstrapped"
        conn = kb.connect()
        try:
            task_id = ready_task(conn, title="legacy work")
            assert kb.claim_task(conn, task_id) is not None
        finally:
            conn.close()
        assert task_row(legacy, task_id)["status"] == "running"

    _control_board_still_works("control")


# ---------------------------------------------------------------------------
# The write chokepoint — a removal in flight, and a handle that outlives one
# ---------------------------------------------------------------------------

def test_a_removal_racing_a_live_handle_refuses_the_claim(fence_home):
    """The handle was admitted BEFORE the fence closed; the claim is not.

    A worker holding an open connection is the exact shape a removal
    races: nothing re-opens, so a check at open time would never fire.
    """
    create_fenced_board("racing")
    create_fenced_board("control")
    db_path = kb.kanban_db_path(board="racing")
    conn = kb.connect(board="racing")
    task_id = ready_task(conn)

    # The removal reaches its fence-closing point under the live handle.
    close_fence("racing")

    try:
        with pytest.raises(kb.BoardFenceClosedError) as excinfo:
            kb.claim_task(conn, task_id)
    finally:
        conn.close()

    assert excinfo.value.refusal.rule is kb.FenceRefusalRule.CLOSED
    assert task_row(db_path, task_id)["status"] == "ready", "the claim landed"
    _control_board_still_works("control")


def test_a_handle_bound_to_a_removed_board_can_never_write(fence_home):
    """The removal completes under an open handle; the handle is dead.

    Permanent mode destroys the storage outright, so this also proves the
    refused write does not bring the board's directory back.
    """
    create_fenced_board("outlived")
    create_fenced_board("control")
    conn = kb.connect(board="outlived")
    task_id = ready_task(conn)

    result = kb.remove_board_fenced(
        "outlived",
        mode="permanent",
        permanent_confirmation=permanent_confirmation("outlived"),
    )
    assert result.success, result.message
    assert not kb.board_dir("outlived").exists()

    try:
        with pytest.raises(kb.BoardFenceClosedError) as excinfo:
            kb.claim_task(conn, task_id)
    finally:
        conn.close()

    assert excinfo.value.refusal.rule is kb.FenceRefusalRule.CLOSED
    assert register_row("outlived")["lifecycle"] == "hard-removed"
    assert not kb.board_dir("outlived").exists(), "the write re-created the board"
    _control_board_still_works("control")


def test_restoring_a_pre_removal_copy_does_not_make_the_board_writable(
    fence_home, tmp_path,
):
    """A stale epoch is refused even when the bytes are all back in place.

    Putting the board's pre-removal storage back at its path is the
    hands-on version of resurrecting it: the files exist, the schema is
    intact, the in-board gate reads ``open``. The register has moved on,
    so the restored store's epoch mirror is stale and EM-4c refuses.
    """
    create_fenced_board("restored")
    create_fenced_board("control")
    db_path = kb.kanban_db_path(board="restored")
    conn = kb.connect(board="restored")
    task_id = ready_task(conn)
    conn.close()

    pre_removal = tmp_path / "pre-removal-copy"
    shutil.copytree(kb.board_dir("restored"), pre_removal)

    intent = start_removal("restored")
    advance = kb.advance_removal_to_fenced(
        "restored", removal_id=intent.removal_id
    )
    assert advance.success, advance.message

    # The operator puts the pre-removal bytes back.
    shutil.rmtree(kb.board_dir("restored"))
    shutil.copytree(pre_removal, kb.board_dir("restored"))
    kb._INITIALIZED_PATHS.clear()

    conn = kb.connect(board="restored")
    try:
        with pytest.raises(kb.BoardFenceClosedError) as excinfo:
            kb.claim_task(conn, task_id)
    finally:
        conn.close()

    assert excinfo.value.refusal.rule is kb.FenceRefusalRule.EM_4c
    assert task_row(db_path, task_id)["status"] == "ready"
    _control_board_still_works("control")


# ---------------------------------------------------------------------------
# The shared init-lock helper's boundary
#
# ``kanban_db._cross_process_init_lock`` delegates to the helper the projects
# store shares, ``sqlite_util.cross_process_init_lock``. That helper used to
# ``mkdir(parents=True, exist_ok=True)`` unconditionally, so the kanban
# caller's "is the directory still there?" check was a check/use race: a
# removal landing in the gap was not refused at the filesystem at all — the
# helper re-created the removed board's directory and put ``kanban.db``'s
# sibling ``init.lock`` inside it, and the board was discoverable again.
#
# These drive the window ONE REAL CALL DEEPER than the tests above: the hook
# is on ``_shared_cross_process_init_lock``, the helper itself, so the
# refusal being asserted is the one the helper's own ``open`` produced and
# not one the kanban wrapper reached before it.
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _removal_lands_inside_the_shared_helper(home: Path, slug: str):
    """Run a REAL removal INSIDE the shared helper's call, not before it.

    The kanban wrapper has already decided the directory is there and has
    already handed the path to the helper; everything from here on is the
    helper's own filesystem work. A removal that lands at this instant is
    the case a pre-check cannot cover, so this is where "the helper never
    materialises what it was asked to lock" has to hold.

    ``**kwargs`` is forwarded verbatim: the kanban caller passes its
    bounded-acquire parameters and its no-create opt-in through here, and a
    hook that dropped them would be testing a different call than the one
    production makes.
    """
    path = kb.kanban_db_path(board=slug)
    original = kb._shared_cross_process_init_lock
    reached: list = []
    kb._INITIALIZED_PATHS.clear()

    @contextlib.contextmanager
    def remove_then_lock(target, **kwargs):
        if Path(target) == path and not reached:
            reached.append(True)
            _remove_in_another_process(home, slug)
            assert not path.parent.exists(), "the removal did not take the board"
        with original(target, **kwargs):
            yield

    kb._shared_cross_process_init_lock = remove_then_lock
    try:
        yield reached
    finally:
        kb._shared_cross_process_init_lock = original


@contextlib.contextmanager
def _watch_the_shared_helper():
    """Record every path the shared helper is asked to place a lock beside.

    Nothing is removed here: this is for asserting which paths the helper
    is ASKED about at all, which is how "the cached fast path never even
    reaches it" becomes an assertion rather than an inference.
    """
    original = kb._shared_cross_process_init_lock
    asked: list = []

    @contextlib.contextmanager
    def record(target, **kwargs):
        asked.append(Path(target))
        with original(target, **kwargs):
            yield

    kb._shared_cross_process_init_lock = record
    try:
        yield asked
    finally:
        kb._shared_cross_process_init_lock = original


def test_the_shared_helper_refuses_an_absent_board_directory(fence_home):
    """The kanban caller's no-create opt-in, at the helper itself.

    Straight at the boundary: the board directory is gone, and taking the
    init lock for a store inside it must refuse instead of building the
    directory back to hold the lock file.
    """
    create_fenced_board("helper-direct")
    create_fenced_board("control")
    db_path = kb.kanban_db_path(board="helper-direct")
    shutil.rmtree(kb.board_dir("helper-direct"))

    with pytest.raises(kb.BoardFenceClosedError) as excinfo:
        with kb._cross_process_init_lock(db_path):
            pass

    assert excinfo.value.refusal.rule is kb.FenceRefusalRule.GA_4
    _assert_left_nothing_behind("helper-direct", what="the init lock")

    # The same refusal, one layer down, in the helper's own vocabulary.
    with pytest.raises(InitLockDirectoryAbsent):
        with cross_process_init_lock(db_path, require_existing_directory=True):
            pass
    _assert_left_nothing_behind("helper-direct", what="the shared helper")
    _control_board_still_works("control")


def test_the_shared_helper_still_creates_the_directory_by_default(tmp_path):
    """The opt-in is opt-IN: the default contract is unchanged.

    ``projects_db`` and the board REGISTER's own lock (which lives outside
    every board, precisely so it survives a board's removal) both rely on
    the helper making their directory. A fix that closed the boundary by
    changing the default would have broken them instead.
    """
    store = tmp_path / "made" / "by" / "the" / "helper" / "projects.db"
    assert not store.parent.exists()

    with cross_process_init_lock(store):
        pass

    assert store.parent.is_dir(), "the default caller lost its directory"
    assert store.with_name(store.name + ".init.lock").exists()


def test_a_cold_open_refuses_a_removal_that_lands_inside_the_helper(fence_home):
    """Cold cache, real second process, removal INSIDE the helper's call.

    ``connect()`` observed a store that was really there, so nothing it
    decided was wrong when it decided it. The removal lands after that and
    after the wrapper's own check, with the helper already holding the
    path — and the helper is what must refuse.
    """
    create_fenced_board("cold-helper")
    create_fenced_board("control")
    conn = kb.connect(board="cold-helper")
    ready_task(conn)
    conn.close()
    kb._INITIALIZED_PATHS.clear()

    with _removal_lands_inside_the_shared_helper(fence_home, "cold-helper") as reached:
        with pytest.raises(kb.BoardFenceClosedError) as excinfo:
            kb.connect(board="cold-helper").close()

    assert reached, "the removal never landed inside the helper"
    assert excinfo.value.refusal.rule is kb.FenceRefusalRule.GA_4
    _assert_left_nothing_behind("cold-helper", what="the refused cold open")
    _control_board_still_works("control")


def test_explicit_init_refuses_a_removal_that_lands_inside_the_helper(fence_home):
    """The explicit creation path, losing the same race one layer deeper.

    ``init_db`` is the one route that asks for ``create=True``, and it is
    what every ``hermes kanban <anything>`` auto-init runs. Admission was
    granted before the removal existed; the helper must still not be the
    thing that puts the board's directory and its lock file back.
    """
    create_fenced_board("init-helper")
    create_fenced_board("control")
    conn = kb.connect(board="init-helper")
    ready_task(conn)
    conn.close()

    with _removal_lands_inside_the_shared_helper(fence_home, "init-helper") as reached:
        with pytest.raises(kb.BoardFenceClosedError) as excinfo:
            kb.init_db(board="init-helper")

    assert reached, "the removal never landed inside the helper"
    assert excinfo.value.refusal.rule is kb.FenceRefusalRule.GA_4
    _assert_left_nothing_behind("init-helper", what="the refused auto-init")
    _control_board_still_works("control")


def test_the_cached_fast_path_never_asks_the_helper_for_a_removed_board(
    fence_home,
):
    """The fast path must refuse without the helper being involved at all.

    This process's ``_INITIALIZED_PATHS`` still says the path is
    initialized — the removal ran in a REAL second process, so nothing
    here could have learned otherwise. That is a long-lived gateway or
    dashboard's exact state. The open must be refused by the store itself,
    and the helper must never be handed the removed board's path: being
    asked at all is what used to re-create the directory.
    """
    create_fenced_board("cached-helper")
    create_fenced_board("control")
    conn = kb.connect(board="cached-helper")
    ready_task(conn)
    conn.close()
    db_path = kb.kanban_db_path(board="cached-helper")
    assert str(db_path.resolve()) in kb._INITIALIZED_PATHS

    _remove_in_another_process(fence_home, "cached-helper")
    assert str(db_path.resolve()) in kb._INITIALIZED_PATHS, (
        "this process must still believe the path is initialized, or the "
        "cached fast path is not the one under test"
    )

    with _watch_the_shared_helper() as asked:
        with pytest.raises(kb.BoardFenceClosedError) as excinfo:
            kb.connect(board="cached-helper")

    assert excinfo.value.refusal.rule is kb.FenceRefusalRule.GA_4
    assert db_path not in asked, (
        f"the cached fast path handed the removed board to the init-lock "
        f"helper: {asked}"
    )
    _assert_left_nothing_behind("cached-helper", what="the cached open")
    _control_board_still_works("control")


def test_a_stale_handle_survives_a_real_removal_without_reviving_it(fence_home):
    """A handle that outlives a removal, and the reopen that follows it.

    The worker's connection was opened while the board was live and is
    never re-opened, so no check at open time can fire for it. Its write
    is refused at the chokepoint; the reopen its caller then attempts is
    refused at the door; and between them nothing — no directory, no
    store, no ``init.lock``, no ``board.json`` — comes back.
    """
    create_fenced_board("stale-handle")
    create_fenced_board("control")
    conn = kb.connect(board="stale-handle")
    task_id = ready_task(conn)

    _remove_in_another_process(fence_home, "stale-handle")
    assert not kb.board_dir("stale-handle").exists()

    try:
        with pytest.raises(kb.BoardFenceClosedError) as write_refusal:
            kb.claim_task(conn, task_id)
    finally:
        conn.close()
    assert write_refusal.value.refusal.rule is kb.FenceRefusalRule.CLOSED

    # The caller does what a worker does next: reconnect and retry.
    with pytest.raises(kb.BoardFenceClosedError) as reopen_refusal:
        kb.connect(board="stale-handle")
    assert reopen_refusal.value.refusal.rule is kb.FenceRefusalRule.GA_4

    _assert_left_nothing_behind("stale-handle", what="the stale handle")
    _control_board_still_works("control")


def test_a_brand_new_board_is_still_created_while_a_removal_is_in_flight(
    fence_home,
):
    """Serializing creation against removal must not break creation.

    Creation now runs under the board's register lock, which is the lock a
    removal takes too — so "no board can be created any more" would satisfy
    every assertion above. A REAL removal of one board runs in a second
    process, and a genuinely new board is created here, through
    ``create_board``, and is fully usable afterwards.

    The handshake is deliberately NOT circular: this process holds no
    register lock while the child runs, and the two boards' locks are
    distinct files, so neither side can be waiting on the other.
    """
    create_fenced_board("doomed-neighbour")
    create_fenced_board("control")

    _remove_in_another_process(fence_home, "doomed-neighbour")
    _assert_left_nothing_behind("doomed-neighbour", what="the removal")

    meta = kb.create_board("born-alongside")
    assert meta["slug"] == "born-alongside"
    assert kb.kanban_db_path(board="born-alongside").exists()
    _control_board_still_works("born-alongside")
    _control_board_still_works("control")
    assert all(entry["slug"] != "doomed-neighbour" for entry in kb.list_boards())
