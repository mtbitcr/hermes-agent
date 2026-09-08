"""A destructive apply whose journal does not commit BLOCKS (§7.2).

The apply journal is the only durable record of what a permanent removal
has already done to the world. A journal write that fails and is stepped
over leaves the next destructive or deregistration step running with no
durable account of the previous one — and, at the end, a receipt built
from an empty journal that reads as a clean pass over real, irreversible
destruction.

Every test here drives a REAL surface — the shipped ``boards rm`` command
through its real argparse tree and dispatch, or the shipped dashboard
``DELETE /boards/{slug}`` route through a real ASGI client — against a
REAL temporary git repository with a REAL linked work area, and reads the
durable outcome back with a plain read-only SQLite connection.

What is being pinned down:

* A failed journal commit stops the sequence where it stands. Nothing
  terminal is recorded, no receipt is written, and the board stays
  resumable.
* The blocking check sits BETWEEN a work area's content going and its
  registration going — the ordering that keeps a crash from orphaning
  content whose owner is no longer identifiable.
* Before leaving Applied, the journal must ACCOUNT FOR every member the
  carried ledger says was to be acted on. A short journal blocks.
* A real SIGKILL right after deregistration, then a restart through the
  shipped entry point, yields exactly ONE journal entry and ONE receipt
  destruction record per resource.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import signal
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

_WORKTREE = Path(__file__).resolve().parents[2]
if str(_WORKTREE) not in sys.path:
    sys.path.insert(0, str(_WORKTREE))

from hermes_cli import kanban_db as kb
from tests.hermes_cli._kanban_fence_support import (
    board_with_multi_commit_advance,
    read_only,
    register_row,
)

_CHILD = Path(__file__).parent / "_kanban_removal_crash_child.py"


# ---------------------------------------------------------------------------
# Real surfaces, loaded the way production loads them
# ---------------------------------------------------------------------------

def _run_cli(argv: list) -> "tuple[int, str]":
    """Run ``hermes kanban …`` through the real argparse tree and dispatch."""
    from hermes_cli import kanban as kanban_cli

    wrap = argparse.ArgumentParser(prog="hermes-test", add_help=False)
    top = wrap.add_subparsers(dest="_top")
    parser = kanban_cli.build_parser(top)
    args = parser.parse_args(argv)
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
        code = kanban_cli.kanban_command(args)
    return code, buffer.getvalue()


def _remove_via_cli(slug: str) -> "tuple[int, str]":
    """The real permanent-removal command, with the statement-bound line."""
    required = kb.permanent_removal_disclosure(slug).required_response
    return _run_cli(["boards", "rm", slug, "--delete", "--confirm", required])


def _dashboard_client():
    """A real ASGI client over the shipped kanban dashboard router."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    plugin_file = (
        _WORKTREE / "plugins" / "kanban" / "dashboard" / "plugin_api.py"
    )
    spec = importlib.util.spec_from_file_location(
        "hermes_kanban_plugin_journal_test", plugin_file
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    app = FastAPI()
    app.include_router(module.router, prefix="/api/plugins/kanban")
    return TestClient(app)


def _remove_via_route(client, slug: str):
    """The real two-step route flow: 409 disclosure, then the bound line."""
    first = client.delete(f"/api/plugins/kanban/boards/{slug}?delete=true")
    assert first.status_code == 409, first.text
    disclosure = first.json()["detail"]["disclosure"]
    return client.request(
        "DELETE",
        f"/api/plugins/kanban/boards/{slug}?delete=true",
        json={
            "confirm": disclosure["required_response"],
            "statement_digest": disclosure["statement_digest"],
        },
    )


# ---------------------------------------------------------------------------
# Durable readers — never through the module under test
# ---------------------------------------------------------------------------

def _journal_items(slug: str) -> list:
    with read_only(kb.register_db_path()) as conn:
        row = conn.execute(
            "SELECT apply_journal FROM board_removal_phase WHERE board_name = ?",
            (slug,),
        ).fetchone()
    if row is None or not row["apply_journal"]:
        return []
    return json.loads(row["apply_journal"])["items"]


def _phase(slug: str) -> str:
    with read_only(kb.register_db_path()) as conn:
        row = conn.execute(
            "SELECT phase FROM board_removal_phase WHERE board_name = ?",
            (slug,),
        ).fetchone()
    assert row is not None, f"{slug} has no durable removal phase record"
    return row["phase"]


def _receipt_rows(slug: str) -> list:
    path = kb.archive_db_path()
    if not path.exists():
        return []
    with read_only(path) as conn:
        return [
            dict(row) for row in conn.execute(
                "SELECT removal_id, status, receipt_payload "
                "FROM board_removal_receipt WHERE board_name = ?",
                (slug,),
            )
        ]


def _assert_nothing_terminal(slug: str) -> None:
    """No terminal lifecycle, no terminal phase, no receipt of any kind."""
    assert register_row(slug)["lifecycle"] == kb.BoardLifecycle.REMOVING.value
    assert _phase(slug) != kb.RemovalPhase.DONE.value
    assert _receipt_rows(slug) == [], (
        "a receipt accounted for a removal that was blocked"
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def board(fence_home, tmp_path):
    """A fenced board with a real repository and a real linked work area."""
    return board_with_multi_commit_advance(
        "journal-fence", tmp_path / "repo", extra_commits=2,
    )


@contextlib.contextmanager
def _failing_journal(monkeypatch, *, only_step=None, silently_skip=None):
    """Replace the REAL journal seam so a named write does not commit.

    ``only_step`` fails just that step, so the blocking check can be
    observed at one exact point in the sequence. ``silently_skip`` is the
    nastier shape: the write reports success and records NOTHING, which is
    what a completeness check over the carried ledger exists to catch.

    A nested ``monkeypatch.context()``, never ``monkeypatch.undo()``: undo
    would also revert the ``fence_home`` fixture's own HERMES_HOME patch
    and point the rest of the test at a different install.
    """
    real = kb.journal_apply_item

    def _patched(slug, *, removal_id, phase, item):
        step = item.get("step")
        if silently_skip is not None:
            if step == silently_skip:
                return True
            return real(slug, removal_id=removal_id, phase=phase, item=item)
        if only_step is None or step == only_step:
            return False
        return real(slug, removal_id=removal_id, phase=phase, item=item)

    with monkeypatch.context() as journal_down:
        journal_down.setattr(kb, "journal_apply_item", _patched)
        yield


# ---------------------------------------------------------------------------
# A failed journal write blocks, through both real entry points
# ---------------------------------------------------------------------------

def test_cli_a_failed_journal_write_blocks_the_whole_apply(board, monkeypatch):
    """`boards rm --delete` refuses, and claims nothing terminal."""
    slug = board["slug"]
    with _failing_journal(monkeypatch):
        code, output = _remove_via_cli(slug)

    assert code == 1, output
    _assert_nothing_terminal(slug)
    assert _journal_items(slug) == [], (
        "the journal recorded items the seam reported as not committed"
    )
    # The sequence stopped at its FIRST journal write, so nothing past the
    # board's own storage was touched: the owned work area and the shared
    # container's registration entry for it are both still there.
    assert board["work_area"].exists(), (
        "the apply went on destroying after a journal write did not commit"
    )
    assert board["registration"].exists()


def test_route_a_failed_journal_write_blocks_the_whole_apply(
    board, monkeypatch
):
    """The dashboard route refuses on the same durable facts."""
    slug = board["slug"]
    client = _dashboard_client()

    with _failing_journal(monkeypatch):
        response = _remove_via_route(client, slug)

    assert response.status_code == 400, response.text
    detail = response.json()["detail"]
    assert detail["message"], "the route refused without saying why"
    _assert_nothing_terminal(slug)
    assert _journal_items(slug) == []
    assert board["work_area"].exists()
    assert board["registration"].exists()


def test_a_failed_journal_write_blocks_between_destruction_and_deregistration(
    board, monkeypatch
):
    """The exact window the ordering exists to protect.

    A work area's CONTENT is destroyed before its registration, because
    the registration is the only durable thing that establishes who owns
    the content. So the blocking check has to sit between the two: with
    the destruction journalled and the entry not committed, the
    registration must still be there for a restart to identify.
    """
    slug = board["slug"]
    with _failing_journal(
        monkeypatch, only_step=kb.APPLY_JOURNAL_WORK_AREA_DESTROYED,
    ):
        code, output = _remove_via_cli(slug)

    assert code == 1, output
    _assert_nothing_terminal(slug)
    # The content really did go — this is the destructive step whose
    # journal entry was lost, not a step that never ran.
    assert not board["work_area"].exists()
    # …and the metadata that says who owned it did NOT.
    assert board["registration"].exists(), (
        "the registration was removed after its work area's journal entry "
        "failed to commit: nothing durable now identifies that content's owner"
    )
    steps = {item["step"] for item in _journal_items(slug)}
    assert steps, "no journal item survived at all"
    assert kb.APPLY_JOURNAL_DEREGISTERED not in steps


def test_a_journal_that_reports_success_without_recording_blocks_applied(
    board, monkeypatch
):
    """A short journal is caught against the carried ledger before Done.

    Every step here reports success, so the apply's own failure list is
    empty. What blocks is the completeness check: the carried ledger names
    a member the durable journal has no entry for.
    """
    slug = board["slug"]
    with _failing_journal(
        monkeypatch, silently_skip=kb.APPLY_JOURNAL_WORK_AREA_DESTROYED,
    ):
        code, output = _remove_via_cli(slug)

    assert code == 1, output
    _assert_nothing_terminal(slug)
    items = _journal_items(slug)
    assert items, "the journal is empty, so this proves nothing about a SHORT one"
    steps = {item["step"] for item in items}
    assert kb.APPLY_JOURNAL_WORK_AREA_DESTROYED not in steps
    record = kb.get_removal_phase_record(slug)
    missing = kb.missing_apply_journal_entries(slug, record)
    assert missing, "the completeness check found nothing missing"
    assert {entry["action"] for entry in missing} == {
        kb.APPLY_JOURNAL_WORK_AREA_DESTROYED
    }


def test_the_completeness_check_covers_every_member_the_carry_ledgered(board):
    """Expectations come from the carried ledger, not from the journal."""
    slug = board["slug"]
    intent = kb.record_removal_intent(
        slug, mode="permanent",
        permanent_confirmation=kb.confirm_permanent_removal(
            slug,
            response=kb.permanent_removal_disclosure(slug).required_response,
            confirmed_by="operator",
        ).confirmation,
    )
    assert intent.success, intent.message
    for driver in (
        kb.advance_removal_to_fenced,
        kb.advance_removal_to_quiesced,
        kb.advance_removal_to_carried,
    ):
        step = driver(slug, removal_id=intent.removal_id)
        assert step.success, f"{driver.__name__}: {step.message}"

    record = kb.get_removal_phase_record(slug)
    payload = record.carried()
    assert payload is not None, "nothing was carried"
    expected = kb.carried_apply_expectations(slug, payload)

    assert expected, "the carried ledger produced no expectations at all"
    actions = {action for action, _identity in expected}
    for required in (
        kb.APPLY_JOURNAL_STORAGE_DESTROYED,
        kb.APPLY_JOURNAL_WORK_AREA_DESTROYED,
        kb.APPLY_JOURNAL_DEREGISTERED,
        kb.APPLY_JOURNAL_RETENTION_RECORDED,
    ):
        assert required in actions, f"{required} is not expected of any member"
    identities = {identity for _action, identity in expected}
    assert str(board["work_area"]) in identities
    assert str(board["registration"]) in identities
    # Nothing has been applied yet, so every one of them is missing.
    missing = kb.missing_apply_journal_entries(slug, record)
    assert len(missing) == len(expected)


# ---------------------------------------------------------------------------
# A blocked apply is resumable, and resuming does not duplicate
# ---------------------------------------------------------------------------

def test_a_blocked_apply_finishes_once_the_journal_commits_again(
    board, monkeypatch
):
    """The same shipped command, run again, completes exactly once."""
    slug = board["slug"]
    with _failing_journal(
        monkeypatch, only_step=kb.APPLY_JOURNAL_WORK_AREA_DESTROYED,
    ):
        blocked, output = _remove_via_cli(slug)
        assert blocked == 1, output
        _assert_nothing_terminal(slug)

    # The seam is back to the real one; the SAME shipped command re-runs.
    code, output = _remove_via_cli(slug)

    assert code == 0, output
    assert register_row(slug)["lifecycle"] == (
        kb.BoardLifecycle.HARD_REMOVED.value
    )
    _assert_one_record_per_resource(slug, board)


def _assert_one_record_per_resource(slug: str, board: dict) -> None:
    """Exactly one durable journal entry and one receipt record per resource."""
    items = _journal_items(slug)
    assert items, "the journal is empty over a completed destructive apply"
    successes = [item for item in items if item.get("ok")]
    assert successes, "no journal item records anything as done"
    duplicates = [
        key for key, count in
        Counter((item["step"], item["identity"]) for item in successes).items()
        if count > 1
    ]
    assert duplicates == [], f"one resource, several journal entries: {duplicates}"

    identities = {(item["step"], item["identity"]) for item in successes}
    assert (
        kb.APPLY_JOURNAL_WORK_AREA_DESTROYED, str(board["work_area"])
    ) in identities
    assert (
        kb.APPLY_JOURNAL_DEREGISTERED, str(board["registration"])
    ) in identities

    rows = _receipt_rows(slug)
    assert len(rows) == 1, rows
    assert rows[0]["status"] == kb.RECEIPT_STATUS_APPLIED
    receipt = json.loads(rows[0]["receipt_payload"])
    records = receipt["destroyed"]["records"]
    assert records, "the receipt records nothing as destroyed"
    record_duplicates = [
        key for key, count in
        Counter(
            (item.get("step"), item.get("identity")) for item in records
        ).items()
        if count > 1
    ]
    assert record_duplicates == [], (
        f"one resource, several receipt destruction records: {record_duplicates}"
    )
    destroyed_identities = {item.get("identity") for item in records}
    assert str(board["work_area"]) in destroyed_identities


# ---------------------------------------------------------------------------
# A real SIGKILL, a real restart, and still one record per resource
# ---------------------------------------------------------------------------

def _run_child(home: Path, slug: str, crash: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(_CHILD), str(home), slug, crash, "permanent"],
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


def test_a_kill_right_after_deregistration_leaves_one_record_per_resource(
    fence_home, tmp_path
):
    """Crash after the registration goes, restart, and count what is durable.

    An uncatchable SIGKILL immediately after the deregistration is the
    exact window where a re-run re-performs already-completed work. The
    steps are idempotent on the filesystem — destroying what is absent
    succeeds — so without an exact-identity check the journal grows a
    SECOND entry for the same work area, and the receipt a SECOND
    destruction record.
    """
    board = board_with_multi_commit_advance(
        "journal-crash", tmp_path / "repo", extra_commits=2,
    )
    slug = board["slug"]

    crashed = _run_child(fence_home, slug, "apply-after-deregister")

    assert crashed.returncode == -signal.SIGKILL, (
        f"the child did not die of SIGKILL: rc={crashed.returncode} "
        f"out={crashed.stdout} err={crashed.stderr}"
    )
    # The window is real: the content and its registration are both gone
    # and the removal has recorded nothing terminal.
    assert not board["work_area"].exists()
    assert not board["registration"].exists()
    _assert_nothing_terminal(slug)
    before = [item for item in _journal_items(slug) if item.get("ok")]
    assert before, "nothing was journalled before the crash"

    restarted = _run_child(fence_home, slug, "never")

    assert restarted.returncode == 0, (
        f"the restart failed: out={restarted.stdout} err={restarted.stderr}"
    )
    assert register_row(slug)["lifecycle"] == (
        kb.BoardLifecycle.HARD_REMOVED.value
    )
    _assert_one_record_per_resource(slug, board)
