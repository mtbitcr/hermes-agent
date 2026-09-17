"""§7.3: a verified archive, its work areas, and a board that stays findable.

One regression each for the four things a reversible removal owes: a
crash MID-COPY costs the live board nothing and restarts clean; a PARTIAL
retained directory is completed against the recorded intent or REFUSED;
the carried C13 work-area ledger goes through the SAME path permanent
mode uses; and the copy records ARCHIVED in its own metadata and appears
in the archived listing before any terminal success.

Durable facts are read through a plain read-only connection, and the
crash is a real SIGKILL in a real subprocess driving the shipped CLI.
"""

from __future__ import annotations

import json
import shutil
import signal
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
    permanent_confirmation, read_only, ready_task, record_git_receipt,
    register_row, start_removal,
)

_CHILD = Path(__file__).parent / "_kanban_removal_crash_child.py"
#: Dies with the copy on disk and the verification not yet run.
_MID_COPY = "apply-mid-retained-copy"
_C13_PATH = (
    "verify_work_area_ownership", "_destroy_work_area_content",
    "_deregister_work_area",
)


def _run_child(home: Path, slug: str, crash: str):
    return subprocess.run(
        [sys.executable, str(_CHILD), str(home), slug, crash, "reversible"],
        capture_output=True, text=True, timeout=180, cwd=str(_WORKTREE),
        env={"PATH": "/usr/bin:/bin:/usr/local/bin", "PYTHONPATH": str(_WORKTREE),
             "PYTHONHASHSEED": "0", "TZ": "UTC", "LANG": "C.UTF-8",
             "HOME": str(home)},
    )


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


def _board(slug: str) -> list:
    create_fenced_board(slug)
    conn = kb.connect(board=slug)
    ready_task(conn, title="work an archive has to keep")
    conn.close()
    return _tasks(kb.kanban_db_path(board=slug))


def _applied(slug: str) -> str:
    """A real reversible removal driven to Applied, with one real task."""
    _board(slug)
    intent = start_removal(slug, mode="reversible")
    assert intent.success, intent.message
    for step in (kb.advance_removal_to_fenced, kb.advance_removal_to_quiesced,
                 kb.advance_removal_to_carried, kb.advance_removal_to_released,
                 kb.advance_removal_to_applied):
        result = step(slug, removal_id=intent.removal_id)
        assert result.success, f"{step.__name__}: {result.message}"
    return intent.removal_id


def _with_work_area(slug: str, repo: Path) -> dict:
    """A fenced board whose durable task row names a REAL linked work area."""
    base = make_git_repo(repo)
    create_fenced_board(slug)
    conn = kb.connect(board=slug)
    task = ready_task(conn, title="work in a worktree")
    work_area, head = add_linked_work_area(repo, task, branch=f"hermes/{task}")
    record_git_receipt(conn, task, workspace_path=work_area,
                       branch_name=f"hermes/{task}", base_commit=base,
                       head_commit=head)
    conn.close()
    return {"task": task, "repo": repo, "work_area": work_area, "head": head,
            "registration": repo / ".git" / "worktrees" / task}


def test_a_crash_mid_copy_keeps_the_live_board_and_restarts_clean(fence_home):
    """Nothing is deleted on the strength of a copy nobody verified."""
    slug = "retained-crash"
    tasks = _board(slug)

    crashed = _run_child(fence_home, slug, _MID_COPY)

    assert crashed.returncode == -signal.SIGKILL, crashed.stderr
    # The live board and every task in it are still there and readable.
    assert kb.board_dir(slug).exists()
    assert _tasks(kb.kanban_db_path(board=slug)) == tasks
    assert register_row(slug)["lifecycle"] == kb.BoardLifecycle.REMOVING.value
    done = {item["step"] for item in _journal(slug) if item.get("ok")}
    assert kb.APPLY_JOURNAL_RETAINED_COPY not in done
    assert kb.APPLY_JOURNAL_STORAGE_DESTROYED not in done

    restarted = _run_child(fence_home, slug, "never")

    assert restarted.returncode == 0, restarted.stderr
    row = _row(slug)
    assert (row["phase"], row["outcome"]) == ("done", "archived")
    assert register_row(slug)["lifecycle"] == kb.BoardLifecycle.ARCHIVED.value
    retained = kb.reversible_retained_path(slug, row["removal_id"])
    # ONE copy at the derived identity, holding exactly the board's data.
    assert sorted(p.name for p in retained.parent.iterdir()) == [retained.name]
    assert _tasks(retained / "kanban.db") == tasks
    assert not kb.board_dir(slug).exists()
    assert not kb.kanban_db_path(board=slug).exists()


def test_a_partial_retained_copy_is_completed_or_the_removal_is_refused(
    fence_home
):
    """Half a copy is finished against the recorded intent — or refused."""
    done, refused, orphan = "part-done", "part-refused", "part-orphan"
    removal, tasks = {}, {}
    for slug in (done, refused, orphan):
        removal[slug] = _applied(slug)
        tasks[slug] = _tasks(kb.kanban_db_path(board=slug))

    # (a) What an interrupted copy leaves: a half-written store, and the
    # board's other files not copied at all.
    retained = kb.reversible_retained_path(done, removal[done])
    retained.mkdir(parents=True)
    (retained / "kanban.db").write_bytes(b"half a database")

    result = kb.apply_reversible_mode_content(done, removal_id=removal[done])

    assert result.success, result.message
    # ONE copy, completed — never a second one alongside the partial.
    assert sorted(p.name for p in retained.parent.iterdir()) == [retained.name]
    assert _tasks(retained / "kanban.db") == tasks[done]
    assert (retained / "board.json").is_file(), "the missing file stayed missing"
    assert not kb.board_dir(done).exists()

    # (b) A copy that cannot be completed: refused, live board untouched.
    (kb.reversible_retained_path(refused, removal[refused]) / "kanban.db").mkdir(
        parents=True)

    result = kb.apply_reversible_mode_content(refused, removal_id=removal[refused])

    assert result.success is False and result.failures
    assert _tasks(kb.kanban_db_path(board=refused)) == tasks[refused]
    items = _journal(refused)
    assert any(not i.get("ok") and i["step"] == kb.APPLY_JOURNAL_RETAINED_COPY
               for i in items), items
    assert not any(i.get("ok") and i["step"] == kb.APPLY_JOURNAL_STORAGE_DESTROYED
                   for i in items)
    assert kb.applied_mode_content_is_outstanding(
        kb.get_removal_phase_record(refused))

    # (c) The shape that must never read as success: partial AND the live
    # original already gone, so nothing can complete it any more.
    copy = kb.reversible_retained_path(orphan, removal[orphan])
    copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(kb.board_dir(orphan), copy)
    (copy / "kanban.db").write_bytes(b"not a database")
    shutil.rmtree(kb.board_dir(orphan))

    result = kb.apply_reversible_mode_content(orphan, removal_id=removal[orphan])

    assert result.success is False and result.failures
    assert kb.applied_mode_content_is_outstanding(
        kb.get_removal_phase_record(orphan))
    assert kb.archived_board_listing_entry(orphan, removal[orphan]) is None
    # A re-drive keeps refusing rather than settling for what is there.
    assert not kb.apply_reversible_mode_content(
        orphan, removal_id=removal[orphan]).success


@pytest.fixture
def watched(monkeypatch):
    """Record every call into the real C13 path, without replacing it."""
    calls: list = []

    def _watch(name):
        real = getattr(kb, name)

        def _wrapper(registration, *args, **kwargs):
            calls.append((name, registration.get("work_area")))
            return real(registration, *args, **kwargs)

        monkeypatch.setattr(kb, name, _wrapper)

    for name in _C13_PATH:
        _watch(name)
    return calls


def test_a_reversible_removal_clears_its_work_areas_the_same_way(
    fence_home, tmp_path, watched
):
    """No directory, no registration, no checkout — through §7.2's own path."""
    rev = _with_work_area("rev-work-area", tmp_path / "repo-a")
    perm = _with_work_area("perm-work-area", tmp_path / "repo-b")
    content = rev["work_area"] / f"{rev['task']}.txt"
    assert content.read_text(encoding="utf-8") == "work\n"

    assert kb.remove_board_fenced("rev-work-area", mode="reversible").success
    reversible = list(watched)
    assert kb.remove_board_fenced(
        "perm-work-area", mode="permanent",
        permanent_confirmation=permanent_confirmation("perm-work-area"),
    ).success

    assert not content.exists(), "the work area's content is still readable"
    assert not rev["work_area"].exists(), "the work area directory survived"
    assert not rev["registration"].exists(), "the registration survived"
    listing = git(rev["repo"], "worktree", "list", "--porcelain").splitlines()
    assert [ln.split(" ", 1)[1] for ln in listing
            if ln.startswith("worktree ")] == [str(rev["repo"])], listing
    # C7: the shared container and the commits it holds are untouched.
    assert git(rev["repo"], "cat-file", "-t", rev["head"]) == "commit"
    # The reversible removal went through the SAME three functions, in the
    # same order, as the permanent one.
    assert reversible == [(n, str(rev["work_area"])) for n in _C13_PATH]
    assert watched[len(reversible):] == [
        (n, str(perm["work_area"])) for n in _C13_PATH]
    steps = [i["step"] for i in _journal("rev-work-area") if i.get("ok")]
    assert steps.index(kb.APPLY_JOURNAL_WORK_AREA_DESTROYED) < steps.index(
        kb.APPLY_JOURNAL_DEREGISTERED), steps


def test_an_unowned_work_area_blocks_a_reversible_removal(fence_home, tmp_path):
    """Ownership is not weakened for archiving: unprovable means untouched."""
    setup = _with_work_area("rev-unowned", tmp_path / "repo")
    # The container's own registration entry — the only durable thing that
    # establishes ownership — is gone, while the directory is still there.
    shutil.rmtree(setup["registration"])

    result = kb.remove_board_fenced("rev-unowned", mode="reversible")

    assert result.success is False, result.message
    assert (setup["work_area"] / f"{setup['task']}.txt").exists()
    record = kb.get_removal_phase_record("rev-unowned")
    assert record.phase is not kb.RemovalPhase.DONE
    assert kb.applied_mode_content_is_outstanding(record)
    assert any(i["identity"] == str(setup["work_area"])
               for i in _journal("rev-unowned") if not i.get("ok"))


def test_the_archive_says_so_itself_and_is_listed_before_any_success(
    fence_home, monkeypatch
):
    """Marker in the copy's own metadata, entry in the listing, then Done.

    And the other direction: with the marker's atomic write failing the
    way a full disk fails, the apply blocks and no terminal state is
    recorded — though the copy verified and the live board is gone.
    """
    _board("archived-board")
    _board("archived-gate")
    assert kb.list_archived_boards() == []
    real_write, full_disk = kb._atomic_write_text, {"on": True}

    def _maybe_write(path, text):
        if full_disk["on"]:
            raise OSError("no space left on device")
        real_write(path, text)

    monkeypatch.setattr(kb, "_atomic_write_text", _maybe_write)
    blocked = kb.remove_board_fenced("archived-gate", mode="reversible")

    assert blocked.success is False, blocked.message
    gate = _row("archived-gate")["removal_id"]
    assert _row("archived-gate")["phase"] != "done"
    assert register_row("archived-gate")["lifecycle"] == (
        kb.BoardLifecycle.REMOVING.value)
    assert kb.archived_board_listing_entry("archived-gate", gate) is None
    # The copy exists but does not claim to be an archive.
    unmarked, = [e for e in kb.list_archived_boards()
                 if Path(e["retained_path"]) == kb.reversible_retained_path(
                     "archived-gate", gate)]
    assert unmarked["archived"] is False

    full_disk["on"] = False
    assert kb.drive_removal("archived-gate", resumed=True).success
    assert kb.remove_board_fenced("archived-board", mode="reversible").success

    for slug in ("archived-gate", "archived-board"):
        row = _row(slug)
        assert row["phase"] == "done"
        assert register_row(slug)["lifecycle"] == kb.BoardLifecycle.ARCHIVED.value
        retained = kb.reversible_retained_path(slug, row["removal_id"])
        meta = json.loads((retained / "board.json").read_text(encoding="utf-8"))
        assert meta["archived"] is True
        marker = meta[kb.RETAINED_ARCHIVED_MARKER_KEY]
        assert marker["removal_id"] == row["removal_id"]
        assert marker["retained_path"] == str(retained) and marker["archived_at"]
        entry = kb.archived_board_listing_entry(slug, row["removal_id"])
        assert entry and Path(entry["retained_path"]) == retained
        assert entry["archived_marker"] == marker
        # Emphatically not a live board — which is why the explicit
        # listing exists — and reachable from the shipped surface.
        assert slug not in {b["slug"] for b in kb.list_boards()}
        assert [e for e in json.loads(cli("boards archived --json"))
                if e["slug"] == slug] == [entry]
        assert slug in cli("boards archived")


@pytest.mark.parametrize("corrupt_payload", [
    pytest.param("{not valid json", id="invalid-json"),
    pytest.param("[1, 2, 3]", id="json-array"),
    pytest.param('"just a string"', id="json-string"),
    pytest.param(None, id="absent"),
    pytest.param("", id="empty"),
])
def test_an_unreadable_carry_refuses_instead_of_acting_on_an_empty_ledger(
    fence_home, tmp_path, corrupt_payload
):
    """§7.3: a carry that cannot be READ is refused — never an empty ledger.

    ``(record.carried() or {}).get(...)`` used to turn every unreadable
    shape (bad JSON, a non-object, absent, empty) into ``{}``, so neither
    the work-area loop nor the unresolved-block branch in
    ``_apply_carried_work_areas`` ever ran, and the apply went on to
    delete the live board while the work area, its directory and its
    worktree registration were all left live.
    """
    slug = "rev-unreadable-carry"
    setup = _with_work_area(slug, tmp_path / "repo")
    intent = start_removal(slug, mode="reversible")
    assert intent.success, intent.message
    for step in (kb.advance_removal_to_fenced, kb.advance_removal_to_quiesced,
                 kb.advance_removal_to_carried, kb.advance_removal_to_released,
                 kb.advance_removal_to_applied):
        result = step(slug, removal_id=intent.removal_id)
        assert result.success, f"{step.__name__}: {result.message}"
    removal_id = intent.removal_id

    # The carry step already ran and recorded a valid payload; corrupt the
    # REAL durable column a re-read of the record goes through, after the
    # fact, exactly as an on-disk corruption would.
    with kb.register_connect() as conn:
        conn.execute(
            "UPDATE board_removal_phase SET carried_payload = ? "
            "WHERE board_name = ?", (corrupt_payload, slug),
        )

    result = kb.drive_removal(slug, resumed=True)

    assert result.success is False, result.message
    assert kb.board_dir(slug).exists()
    assert kb.kanban_db_path(board=slug).exists()
    content = setup["work_area"] / f"{setup['task']}.txt"
    assert content.exists(), "the work area's content did not survive"
    assert setup["work_area"].exists(), "the work area directory did not survive"
    assert setup["registration"].exists(), "the registration did not survive"
    listing = git(setup["repo"], "worktree", "list", "--porcelain").splitlines()
    worktrees = [ln.split(" ", 1)[1] for ln in listing if ln.startswith("worktree ")]
    assert str(setup["work_area"]) in worktrees, listing
    row = _row(slug)
    assert row["phase"] != "done"
    assert row["outcome"] != "archived"
    record = kb.get_removal_phase_record(slug)
    assert record.removal_id == removal_id
    assert kb.applied_mode_content_is_outstanding(record)
