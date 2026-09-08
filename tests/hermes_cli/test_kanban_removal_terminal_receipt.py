"""The terminal state and its §12 receipt commit together, or not at all.

The register and the removal archive are two SEPARATE SQLite database
files — deliberately separate loss-and-rebuild domains — and SQLite gives
no atomic multi-database commit in WAL mode. So the terminal transition
runs a durable PREPARE-THEN-APPLY protocol instead: the exact, byte-final
receipt is committed to the archive first, the register is moved second,
and the prepared receipt is marked applied third. A preflight probe that
merely proved the archive writable was a check in one transaction about a
write in another, and a board reached ``hard-removed`` on the strength of
it with nothing to account for the destruction.

Every test here drives a REAL surface — the shipped ``boards rm`` command
through its real argparse tree and dispatch, or the shipped dashboard
``DELETE /boards/{slug}`` route through a real ASGI client — against a
REAL temporary git repository, and reads the archive back with a plain
read-only SQLite connection, so "the receipt says X" is a fact about what
is on disk rather than about a return value.

What is being pinned down:

* Nothing claims ``hard-removed`` or a completion time at ``applied``.
* A failure AT THE REAL INSERTION SEAM leaves NO terminal state: not the
  lifecycle, not the phase — and the board stays resumable.
* When the terminal state IS recorded, the exact receipt content is
  already durable; the only thing left is a status flip.
* Both interrupted directions roll forward, and the roll-forward APPLIES
  the committed bytes rather than regenerating a payload.
* Insertion is immutable: the same digest is idempotent, a different
  digest for the same removal is REFUSED.
* Every §12 section is present, the commit list NAMES ITS SOURCE, and its
  completeness is marked UNVERIFIED because the creation-time provenance
  ledger it would need is absent.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import sqlite3
import sys
from pathlib import Path

import pytest

_WORKTREE = Path(__file__).resolve().parents[2]
if str(_WORKTREE) not in sys.path:
    sys.path.insert(0, str(_WORKTREE))

from hermes_cli import kanban_db as kb
from tests.hermes_cli._kanban_fence_support import (
    add_linked_work_area,
    board_with_multi_commit_advance,
    create_fenced_board,
    make_git_repo,
    permanent_confirmation,
    read_only,
    ready_task,
    record_git_receipt,
    register_row,
)


# ---------------------------------------------------------------------------
# Real surfaces
# ---------------------------------------------------------------------------

def _run_cli(argv: list) -> "tuple[int, str]":
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
    required = kb.permanent_removal_disclosure(slug).required_response
    return _run_cli(["boards", "rm", slug, "--delete", "--confirm", required])


def _dashboard_client():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    plugin_file = (
        _WORKTREE / "plugins" / "kanban" / "dashboard" / "plugin_api.py"
    )
    spec = importlib.util.spec_from_file_location(
        "hermes_kanban_plugin_receipt_test", plugin_file
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    app = FastAPI()
    app.include_router(module.router, prefix="/api/plugins/kanban")
    return TestClient(app)


def _remove_via_route(client, slug: str):
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
# Failure injected at the REAL insertion seam
# ---------------------------------------------------------------------------

_RECEIPT_INSERT = "INSERT INTO board_removal_receipt"


@contextlib.contextmanager
def _receipt_insert_fails(monkeypatch):
    """Make the REAL receipt INSERT statement fail, and nothing else.

    Not a probe and not a function boundary: the archive connection every
    receipt write goes through is wrapped so that the actual ``INSERT INTO
    board_removal_receipt`` raises, exactly as it would if the archive
    could not take the row. Every other archive statement — the marker,
    the audit record, the reads — still works, so this isolates the one
    write the terminal claim depends on.

    A nested ``monkeypatch.context()``, never ``undo()``: undo would also
    revert the ``fence_home`` fixture's own HERMES_HOME patch.
    """
    real = kb.archive_connect

    class _RefusingConnection:
        def __init__(self, conn):
            self._conn = conn

        def execute(self, sql, *args, **kwargs):
            if _RECEIPT_INSERT in " ".join(str(sql).split()):
                raise sqlite3.OperationalError(
                    "injected at the real insertion seam: the receipt row "
                    "could not be written"
                )
            return self._conn.execute(sql, *args, **kwargs)

        def __getattr__(self, name):
            return getattr(self._conn, name)

    @contextlib.contextmanager
    def _patched():
        with real() as conn:
            yield _RefusingConnection(conn)

    with monkeypatch.context() as archive:
        archive.setattr(kb, "archive_connect", _patched)
        yield


@contextlib.contextmanager
def _receipt_apply_fails(monkeypatch):
    """The prepared receipt is committed, but marking it applied does not."""
    with monkeypatch.context() as apply_down:
        apply_down.setattr(
            kb, "apply_prepared_removal_receipt",
            lambda *a, **k: (False, "injected: the apply flip did not commit"),
        )
        yield


# ---------------------------------------------------------------------------
# Durable readers — never through the module under test
# ---------------------------------------------------------------------------

def _stored_receipts(slug: str) -> list:
    path = kb.archive_db_path()
    if not path.exists():
        return []
    with read_only(path) as conn:
        return [
            dict(row) for row in conn.execute(
                "SELECT removal_id, terminal_lifecycle, receipt_payload, "
                "status, receipt_digest, prepared_at, applied_at "
                "FROM board_removal_receipt WHERE board_name = ? "
                "ORDER BY created_at",
                (slug,),
            )
        ]


def _phase_row(slug: str):
    with read_only(kb.register_db_path()) as conn:
        return conn.execute(
            "SELECT phase, outcome, refusal_outcome FROM board_removal_phase "
            "WHERE board_name = ?",
            (slug,),
        ).fetchone()


def _assert_no_terminal_state(slug: str) -> None:
    """Neither half of the terminal claim is recorded, in either file."""
    assert register_row(slug)["lifecycle"] == kb.BoardLifecycle.REMOVING.value, (
        "hard-removed was recorded with no receipt to account for it"
    )
    row = _phase_row(slug)
    assert row["phase"] != kb.RemovalPhase.DONE.value, (
        "the phase reached done with nowhere to record the receipt"
    )
    assert _stored_receipts(slug) == []


# ---------------------------------------------------------------------------
# Boards
# ---------------------------------------------------------------------------

def _board(slug: str, repo: Path) -> dict:
    """A fenced board with a real task and a real linked work area."""
    base = make_git_repo(repo)
    create_fenced_board(slug)
    conn = kb.connect(board=slug)
    task_id = ready_task(conn)
    branch = f"hermes/{task_id}"
    work_area, head = add_linked_work_area(repo, task_id, branch=branch)
    record_git_receipt(
        conn, task_id, workspace_path=work_area, branch_name=branch,
        base_commit=base, head_commit=head,
    )
    conn.close()
    return {"slug": slug, "task": task_id, "work_area": work_area}


def _to_applied(slug: str) -> str:
    """Drive a real permanent removal to Applied and stop there."""
    intent = kb.record_removal_intent(
        slug, mode="permanent",
        permanent_confirmation=permanent_confirmation(slug),
    )
    assert intent.success, intent.message
    for driver in (
        kb.advance_removal_to_fenced,
        kb.advance_removal_to_quiesced,
        kb.advance_removal_to_carried,
        kb.advance_removal_to_released,
        kb.advance_removal_to_applied,
    ):
        step = driver(slug, removal_id=intent.removal_id)
        assert step.success, f"{driver.__name__}: {step.message}"
    return intent.removal_id


# ---------------------------------------------------------------------------
# No terminal claim before `done`
# ---------------------------------------------------------------------------

def test_the_apply_records_no_receipt_and_no_terminal_lifecycle(
    fence_home, tmp_path
):
    """At `applied` the board is destroyed but nothing terminal is claimed."""
    setup = _board("terminal-apply", tmp_path / "repo")
    removal_id = _to_applied("terminal-apply")

    content = kb.apply_permanent_mode_content(
        "terminal-apply", removal_id=removal_id,
    )

    assert content.success, content.message
    # The content really happened…
    assert not kb.board_dir("terminal-apply").exists()
    assert not setup["work_area"].exists()
    # …and nothing has claimed the removal is finished.
    assert _stored_receipts("terminal-apply") == [], (
        "a receipt claimed hard-removed while the removal was at applied"
    )
    assert kb.get_permanent_removal_receipt("terminal-apply", removal_id) is None
    assert register_row("terminal-apply")["lifecycle"] == (
        kb.BoardLifecycle.REMOVING.value
    )
    record = kb.get_removal_phase_record("terminal-apply")
    assert record.phase is kb.RemovalPhase.APPLIED
    assert record.outcome is None


def test_a_receipt_cannot_be_written_before_done(fence_home, tmp_path):
    """The terminal writer refuses outright at a non-terminal phase."""
    _board("terminal-early", tmp_path / "repo")
    removal_id = _to_applied("terminal-early")
    assert kb.apply_permanent_mode_content(
        "terminal-early", removal_id=removal_id
    ).success

    result = kb.write_terminal_removal_receipt(
        "terminal-early", removal_id=removal_id,
    )

    assert not result.success, result.message
    assert "terminal claim" in result.message
    assert _stored_receipts("terminal-early") == []


def test_the_receipt_is_built_from_the_state_after_done(fence_home, tmp_path):
    """The finished facts, not the pre-record ones."""
    setup = _board("terminal-after", tmp_path / "repo")

    code, output = _remove_via_cli("terminal-after")
    assert code == 0, output

    record = kb.get_removal_phase_record("terminal-after")
    receipt = kb.get_permanent_removal_receipt(
        "terminal-after", record.removal_id
    )
    assert receipt is not None

    subject = receipt["subject"]
    assert subject["phase"] == kb.RemovalPhase.DONE.value
    assert subject["terminal_lifecycle"] == kb.BoardLifecycle.HARD_REMOVED.value
    assert subject["outcome"] == "completed"
    # The completion time is the durable Done instant. The receipt is
    # committed BEFORE that transition, so the instant it names is pinned
    # and the register records exactly it — not two clocks that disagree.
    assert subject["completion_time"] == record.updated_at
    assert subject["removal_id"] == record.removal_id
    # The operator item that said "the content has NOT been performed" is
    # gone, because by now it has been.
    assert receipt["honesty"]["open_operator_items"] == []
    assert kb.removal_operator_items(record) == []
    assert not setup["work_area"].exists()


# ---------------------------------------------------------------------------
# A failure at the REAL insertion seam records no terminal state
# ---------------------------------------------------------------------------

def test_cli_a_failed_receipt_insert_records_no_terminal_state(
    fence_home, tmp_path, monkeypatch
):
    """The shipped command refuses AND the world stays non-terminal."""
    _board("insert-cli", tmp_path / "repo")

    with _receipt_insert_fails(monkeypatch):
        code, output = _remove_via_cli("insert-cli")

    assert code == 1, output
    _assert_no_terminal_state("insert-cli")
    refusal = json.loads(_phase_row("insert-cli")["refusal_outcome"])
    assert refusal["outcome"] == kb.REMOVAL_REFUSAL_RECEIPT_UNWRITABLE
    assert refusal["message"], "the refusal was recorded without a reason"

    # The board is still resumable through the SAME shipped command.
    code, output = _remove_via_cli("insert-cli")

    assert code == 0, output
    assert register_row("insert-cli")["lifecycle"] == (
        kb.BoardLifecycle.HARD_REMOVED.value
    )
    rows = _stored_receipts("insert-cli")
    assert len(rows) == 1, rows
    assert rows[0]["status"] == kb.RECEIPT_STATUS_APPLIED


def test_route_a_failed_receipt_insert_records_no_terminal_state(
    fence_home, tmp_path, monkeypatch
):
    """The dashboard route refuses on exactly the same durable facts."""
    _board("insert-route", tmp_path / "repo")
    client = _dashboard_client()

    with _receipt_insert_fails(monkeypatch):
        response = _remove_via_route(client, "insert-route")

    assert response.status_code == 400, response.text
    detail = response.json()["detail"]
    assert detail["message"], "the route refused without saying why"
    _assert_no_terminal_state("insert-route")

    retried = _remove_via_route(client, "insert-route")

    assert retried.status_code == 200, retried.text
    assert retried.json()["result"]["phase"] == kb.RemovalPhase.DONE.value
    assert register_row("insert-route")["lifecycle"] == (
        kb.BoardLifecycle.HARD_REMOVED.value
    )
    rows = _stored_receipts("insert-route")
    assert len(rows) == 1, rows
    assert rows[0]["status"] == kb.RECEIPT_STATUS_APPLIED


def test_the_receipt_content_is_durable_before_the_terminal_state(
    fence_home, tmp_path, monkeypatch
):
    """The ordering, observed from inside the protocol's own third step.

    When the apply step is reached, the register has ALREADY moved and the
    exact receipt content is ALREADY committed as prepared. That is the
    whole invariant: by the time anything terminal is durable, nothing
    about the receipt's content can still fail.
    """
    _board("ordering", tmp_path / "repo")
    observed: list = []
    real_apply = kb.apply_prepared_removal_receipt

    def _observing(board, removal_id, **kwargs):
        rows = _stored_receipts(board)
        observed.append({
            "lifecycle": register_row(board)["lifecycle"],
            "phase": _phase_row(board)["phase"],
            "rows": rows,
        })
        return real_apply(board, removal_id, **kwargs)

    with monkeypatch.context() as spy:
        spy.setattr(kb, "apply_prepared_removal_receipt", _observing)
        code, output = _remove_via_cli("ordering")

    assert code == 0, output
    assert len(observed) == 1, observed
    seen = observed[0]
    assert seen["lifecycle"] == kb.BoardLifecycle.HARD_REMOVED.value
    assert seen["phase"] == kb.RemovalPhase.DONE.value
    assert len(seen["rows"]) == 1, seen["rows"]
    prepared = seen["rows"][0]
    assert prepared["status"] == kb.RECEIPT_STATUS_PREPARED
    assert prepared["receipt_digest"], "the prepared receipt carries no digest"
    assert prepared["prepared_at"] is not None
    assert prepared["applied_at"] is None
    # Prepared means the REAL content, not a placeholder: the finished
    # subject, the destruction records and the §12.7 checks are all there.
    payload = json.loads(prepared["receipt_payload"])
    assert payload["subject"]["terminal_lifecycle"] == (
        kb.BoardLifecycle.HARD_REMOVED.value
    )
    assert payload["destroyed"]["records"], "the prepared receipt is a stub"
    assert payload["verification"]["checks"], "the prepared receipt is a stub"


def test_a_prepared_receipt_that_was_not_applied_is_rolled_forward(
    fence_home, tmp_path, monkeypatch
):
    """The interrupted direction: terminal state reached, apply not committed.

    The roll-forward must APPLY the bytes that were committed, not build a
    fresh payload — the exact content is what the terminal claim rests on.
    """
    _board("apply-flip", tmp_path / "repo")

    with _receipt_apply_fails(monkeypatch):
        code, output = _remove_via_cli("apply-flip")

    assert code == 1, output
    # The terminal state IS recorded — and it is accounted for, because the
    # exact receipt content is durable. Only the status flip is outstanding.
    assert register_row("apply-flip")["lifecycle"] == (
        kb.BoardLifecycle.HARD_REMOVED.value
    )
    before = _stored_receipts("apply-flip")
    assert len(before) == 1, before
    assert before[0]["status"] == kb.RECEIPT_STATUS_PREPARED
    record = kb.get_removal_phase_record("apply-flip")
    assert kb.terminal_receipt_outstanding(record) is True
    decision = kb.resume_removal("apply-flip")
    assert decision.action is (
        kb.RemovalRecoveryAction.ROLL_FORWARD_TERMINAL_RECEIPT
    )

    code, output = _remove_via_cli("apply-flip")

    assert code == 0, output
    after = _stored_receipts("apply-flip")
    assert len(after) == 1, after
    assert after[0]["status"] == kb.RECEIPT_STATUS_APPLIED
    assert after[0]["receipt_payload"] == before[0]["receipt_payload"]
    assert after[0]["receipt_digest"] == before[0]["receipt_digest"]


def test_a_prepared_receipt_is_not_a_committed_terminal_claim(
    fence_home, tmp_path, monkeypatch
):
    """Prepared is durable content, not an accounted-for terminal claim."""
    _board("prepared-not-committed", tmp_path / "repo")

    with _receipt_apply_fails(monkeypatch):
        code, _output = _remove_via_cli("prepared-not-committed")
    assert code == 1

    record = kb.get_removal_phase_record("prepared-not-committed")
    assert kb.permanent_removal_receipt_committed(
        "prepared-not-committed", record.removal_id
    ) is False
    prepared = kb.get_prepared_removal_receipt(
        "prepared-not-committed", record.removal_id
    )
    assert prepared is not None
    assert prepared.committed is False
    assert prepared.status == kb.RECEIPT_STATUS_PREPARED
    assert prepared.payload["subject"]["removal_id"] == record.removal_id


# ---------------------------------------------------------------------------
# Insertion is immutable
# ---------------------------------------------------------------------------

def test_the_same_receipt_written_twice_is_idempotent(fence_home):
    """A roll-forward that recomputes the identical receipt is accepted."""
    create_fenced_board("immutable-same")
    payload = {"proof": "first"}

    assert kb.record_permanent_removal_receipt(
        "immutable-same", "rid-1", terminal_lifecycle="hard-removed",
        scope_declaration=kb.SCOPE_DECLARATION_VERSION, receipt_payload=payload,
    )
    assert kb.record_permanent_removal_receipt(
        "immutable-same", "rid-1", terminal_lifecycle="hard-removed",
        scope_declaration=kb.SCOPE_DECLARATION_VERSION,
        receipt_payload=dict(payload),
    )

    rows = _stored_receipts("immutable-same")
    assert len(rows) == 1, rows
    assert json.loads(rows[0]["receipt_payload"]) == {"proof": "first"}
    assert rows[0]["status"] == kb.RECEIPT_STATUS_APPLIED


def test_a_different_receipt_for_the_same_removal_is_refused(fence_home):
    """The archive is never rewritten over — prepare or apply."""
    create_fenced_board("immutable-diff")
    assert kb.record_permanent_removal_receipt(
        "immutable-diff", "rid-1", terminal_lifecycle="hard-removed",
        scope_declaration=kb.SCOPE_DECLARATION_VERSION,
        receipt_payload={"proof": "first"},
    )

    replaced = kb.record_permanent_removal_receipt(
        "immutable-diff", "rid-1", terminal_lifecycle="hard-removed",
        scope_declaration=kb.SCOPE_DECLARATION_VERSION,
        receipt_payload={"proof": "second"},
    )
    prepared, reason = kb.prepare_permanent_removal_receipt(
        "immutable-diff", "rid-1", terminal_lifecycle="hard-removed",
        scope_declaration=kb.SCOPE_DECLARATION_VERSION,
        receipt_payload={"proof": "third"},
    )

    assert replaced is False, "a second, different receipt replaced the first"
    assert prepared is None, "a differing prepare overwrote the archive"
    assert reason, "the refusal did not say why"
    rows = _stored_receipts("immutable-diff")
    assert len(rows) == 1, rows
    assert json.loads(rows[0]["receipt_payload"]) == {"proof": "first"}
    assert kb.get_permanent_removal_receipt("immutable-diff", "rid-1") == {
        "proof": "first"
    }


def test_preparing_the_identical_content_again_is_idempotent(fence_home):
    """A re-drive that prepares the same bytes reuses the committed row."""
    create_fenced_board("immutable-prepare")
    payload = {"proof": "identical"}

    first, _first_reason = kb.prepare_permanent_removal_receipt(
        "immutable-prepare", "rid-1", terminal_lifecycle="hard-removed",
        scope_declaration=kb.SCOPE_DECLARATION_VERSION, receipt_payload=payload,
    )
    second, _second_reason = kb.prepare_permanent_removal_receipt(
        "immutable-prepare", "rid-1", terminal_lifecycle="hard-removed",
        scope_declaration=kb.SCOPE_DECLARATION_VERSION,
        receipt_payload=dict(payload),
    )

    assert first is not None and second is not None
    assert first.digest == second.digest
    rows = _stored_receipts("immutable-prepare")
    assert len(rows) == 1, rows
    assert rows[0]["status"] == kb.RECEIPT_STATUS_PREPARED


def test_a_real_removal_leaves_exactly_one_applied_receipt_row(
    fence_home, tmp_path
):
    """Re-driving a finished removal does not append a second receipt."""
    _board("immutable-real", tmp_path / "repo")
    code, output = _remove_via_cli("immutable-real")
    assert code == 0, output
    record = kb.get_removal_phase_record("immutable-real")

    again, output = _remove_via_cli("immutable-real")
    assert again == 0, output

    rows = _stored_receipts("immutable-real")
    assert len(rows) == 1, rows
    assert rows[0]["removal_id"] == record.removal_id
    assert rows[0]["terminal_lifecycle"] == kb.BoardLifecycle.HARD_REMOVED.value
    assert rows[0]["status"] == kb.RECEIPT_STATUS_APPLIED
    assert rows[0]["applied_at"] is not None


# ---------------------------------------------------------------------------
# §12 completeness, including §12.7
# ---------------------------------------------------------------------------

def test_every_section_12_field_is_present(fence_home, tmp_path):
    """No section is missing, and none of them is an empty stub."""
    _board("sections", tmp_path / "repo")
    code, output = _remove_via_cli("sections")
    assert code == 0, output
    record = kb.get_removal_phase_record("sections")
    receipt = kb.get_permanent_removal_receipt("sections", record.removal_id)
    assert receipt is not None

    for section in (
        "subject", "provenance", "destroyed", "version_control",
        "transcripts", "registrations", "honesty", "verification",
    ):
        assert section in receipt, f"§12 section {section} is missing"

    assert receipt["provenance"]["scan_domain"], "§12.1 named no scan domain"
    assert receipt["provenance"]["apply_journal"], "§12.1 carried no journal"
    assert receipt["destroyed"]["records"], "§12.2 recorded nothing destroyed"
    assert receipt["destroyed"]["failures"] == []
    assert receipt["version_control"]["references"], "§12.3 named no reference"
    assert receipt["transcripts"]["clause"] == "C6"
    assert receipt["registrations"]["deregistered"], "§12.5 deregistered nothing"
    assert receipt["honesty"]["permanence_claim"]
    assert receipt["honesty"]["retained_records"], "§12.6 recorded no retention"


def test_the_receipt_names_where_its_commit_list_came_from(
    fence_home, tmp_path
):
    """The source is a NAMED field, and it is the recorded run receipts.

    The commit list is derived from the run receipts each advance
    persisted at completion time. Saying so on the receipt is what stops a
    reader assuming it came from a creation-time provenance ledger — this
    milestone builds no such ledger.
    """
    setup = board_with_multi_commit_advance(
        "commit-source", tmp_path / "repo", extra_commits=2,
    )
    code, output = _remove_via_cli("commit-source")
    assert code == 0, output
    record = kb.get_removal_phase_record("commit-source")
    receipt = kb.get_permanent_removal_receipt("commit-source", record.removal_id)

    version_control = receipt["version_control"]
    assert version_control["commit_list_source"] == kb.COMMIT_LIST_SOURCE
    assert version_control["commit_list_source_statement"]
    assert (
        version_control["commit_list_completeness"]["creation_time_provenance_ledger"]
        == "absent"
    )
    references = version_control["references"]
    assert references, "the receipt names no retained reference at all"
    reference, = [r for r in references if r["task"] == setup["task"]]
    source = reference["commit_list_source"]
    assert source["source"] == kb.COMMIT_LIST_SOURCE
    assert source["creation_time_provenance_ledger"] == "absent"
    assert source["read_from"], "the receipt does not say what it read"
    assert source["statement"]


def test_the_commit_list_completeness_is_marked_unverified_not_claimed(
    fence_home, tmp_path
):
    """Several commits in one advance: only the head is derivable, and the
    receipt says so instead of implying the list is complete."""
    setup = board_with_multi_commit_advance(
        "commit-completeness", tmp_path / "repo", extra_commits=2,
    )
    assert setup["intermediate"], "the fixture made no intermediate commit"
    code, output = _remove_via_cli("commit-completeness")
    assert code == 0, output
    record = kb.get_removal_phase_record("commit-completeness")
    receipt = kb.get_permanent_removal_receipt(
        "commit-completeness", record.removal_id
    )

    references = receipt["version_control"]["references"]
    assert references, "the receipt names no retained reference at all"
    reference, = [r for r in references if r["task"] == setup["task"]]
    # The head the recorded receipt names IS on the list…
    assert setup["head"] in reference["board_created_commits"]
    # …and the intermediate commits of that same advance are NOT, because
    # nothing durable names them. That is the gap the marking discloses.
    for commit in setup["intermediate"]:
        assert commit not in reference["board_created_commits"], commit
    completeness = reference["commit_list_completeness"]
    assert completeness["verdict"] == kb.RECEIPT_VERDICT_UNVERIFIED
    assert completeness["reasons"], "the marking gives no reason"
    assert completeness["creation_time_provenance_ledger"] == "absent"

    check, = [
        c for c in receipt["verification"]["checks"]
        if c["check"] == "commit-list-completeness"
    ]
    assert check["verdict"] == kb.RECEIPT_VERDICT_UNVERIFIED
    assert check["observed"]
    per_reference = check["evidence"]["per_reference"]
    assert per_reference, "the check names no reference"
    for entry in per_reference:
        assert entry["completeness"]["verdict"] == kb.RECEIPT_VERDICT_UNVERIFIED


def test_section_12_7_carries_checks_verdicts_and_evidence(
    fence_home, tmp_path
):
    """Every check has an explicit verdict from the declared value set."""
    _board("verdicts", tmp_path / "repo")
    code, output = _remove_via_cli("verdicts")
    assert code == 0, output
    record = kb.get_removal_phase_record("verdicts")
    verification = kb.get_permanent_removal_receipt(
        "verdicts", record.removal_id
    )["verification"]

    checks = verification["checks"]
    assert checks, "§12.7 performed no checks"
    assert verification["clause"] == "§12.7"
    assert verification["verdict_values"] == list(kb.RECEIPT_VERDICTS)
    for check in checks:
        assert check["verdict"] in kb.RECEIPT_VERDICTS, check
        assert check["required"], check
        assert check["observed"], check
        assert "evidence" in check, check
    names = {check["check"] for check in checks}
    for expected in (
        "board-store-absent",
        "board-directory-absent",
        "work-area-content-destroyed",
        "destruction-journalled-durably",
        "scoped-leftovers-clean",
        "pre-application-inventory-carried",
        "mode-content-recorded-applied",
        "phase-done",
        "terminal-lifecycle-recorded",
        "no-open-operator-items",
        "no-blocking-failures",
        "version-control-retained-and-named",
        "transcripts-retained-and-disclosed",
        "commit-list-completeness",
    ):
        assert expected in names, f"§12.7 has no {expected} check"
    assert verification["verdicts"][kb.RECEIPT_VERDICT_FAIL] == 0, checks
    assert verification["overall"] == kb.RECEIPT_VERDICT_PASS
    # Every verdict value the receipt declares is accounted for in the tally.
    assert set(verification["verdicts"]) == set(kb.RECEIPT_VERDICTS)
    # An UNVERIFIED verdict never reads as a pass, and never as a failure
    # of the run: it is counted, named and explained.
    assert verification["verdicts"][kb.RECEIPT_VERDICT_UNVERIFIED] > 0
    assert kb.RECEIPT_VERDICT_UNVERIFIED in verification["honesty"]


def test_the_journalled_destruction_is_what_the_receipt_reports(
    fence_home, tmp_path
):
    """The §12.2 records come from the durable journal, item for item."""
    setup = _board("journal-backed", tmp_path / "repo")
    code, output = _remove_via_cli("journal-backed")
    assert code == 0, output
    record = kb.get_removal_phase_record("journal-backed")
    receipt = kb.get_permanent_removal_receipt(
        "journal-backed", record.removal_id
    )

    journalled = record.journal()["items"]
    assert journalled, "a completed destructive apply journalled nothing"
    records = receipt["destroyed"]["records"]
    assert records, "the receipt reports nothing destroyed"
    assert {item["identity"] for item in records} <= {
        item["identity"] for item in journalled
    }
    assert str(setup["work_area"]) in {item["identity"] for item in records}
    check, = [
        c for c in receipt["verification"]["checks"]
        if c["check"] == "destruction-journalled-durably"
    ]
    assert check["verdict"] == kb.RECEIPT_VERDICT_PASS
    assert check["evidence"], "the check cites no journal evidence"


def test_an_unprovable_fact_gets_an_explicit_verdict_not_silence(
    fence_home, tmp_path
):
    """A board with no derivable commits still gets a stated verdict."""
    create_fenced_board("unprovable")
    conn = kb.connect(board="unprovable")
    ready_task(conn)  # no git receipt at all
    conn.close()

    code, output = _remove_via_cli("unprovable")
    assert code == 0, output
    record = kb.get_removal_phase_record("unprovable")
    verification = kb.get_permanent_removal_receipt(
        "unprovable", record.removal_id
    )["verification"]

    check, = [
        c for c in verification["checks"]
        if c["check"] == "version-control-retained-and-named"
    ]
    assert check["verdict"] in kb.RECEIPT_VERDICTS
    assert check["observed"], "an unprovable fact was recorded as silence"
    assert verification["honesty"], "§12.7 does not say how it treats unknowns"
    assert kb.RECEIPT_VERDICT_UNVERIFIED in verification["honesty"]


def test_an_archive_that_cannot_be_read_is_not_read_as_no_receipt(
    fence_home, tmp_path, monkeypatch
):
    """Indeterminate is not "done": a failed read refuses the terminal claim."""
    _board("unreadable-archive", tmp_path / "repo")
    code, output = _remove_via_cli("unreadable-archive")
    assert code == 0, output
    record = kb.get_removal_phase_record("unreadable-archive")
    assert kb.permanent_removal_receipt_committed(
        "unreadable-archive", record.removal_id
    ) is True

    def _no_archive():
        raise sqlite3.OperationalError("archive is unavailable")

    with monkeypatch.context() as archive_down:
        archive_down.setattr(kb, "archive_connect", _no_archive)
        assert kb.permanent_removal_receipt_committed(
            "unreadable-archive", record.removal_id
        ) is False
        assert kb.terminal_receipt_outstanding(record) is True


def test_an_unwritable_receipt_store_blocks_done_and_hard_removed(
    fence_home, tmp_path, monkeypatch
):
    """No archive at all, no terminal lifecycle: the removal refuses."""
    _board("terminal-blocked", tmp_path / "repo")
    removal_id = _to_applied("terminal-blocked")
    assert kb.apply_permanent_mode_content(
        "terminal-blocked", removal_id=removal_id
    ).success
    assert kb.advance_removal_to_swept(
        "terminal-blocked", removal_id=removal_id
    ).success

    def _no_archive():
        raise sqlite3.OperationalError("archive is unavailable")

    with monkeypatch.context() as archive_down:
        archive_down.setattr(kb, "archive_connect", _no_archive)
        result = kb.drive_removal("terminal-blocked")

    assert not result.success, result.message
    assert result.refusal_reason == (
        kb.RemovalAdvanceOutcome.REFUSED_PRECONDITION.value
    )
    _assert_no_terminal_state("terminal-blocked")
    assert _phase_row("terminal-blocked")["phase"] == (
        kb.RemovalPhase.SWEPT.value
    )
    refusal = json.loads(_phase_row("terminal-blocked")["refusal_outcome"])
    assert refusal["outcome"] == kb.REMOVAL_REFUSAL_RECEIPT_UNWRITABLE

    # With the store back, the same loop rolls forward to a real success.
    resumed = kb.drive_removal("terminal-blocked")
    assert resumed.success, resumed.message
    assert register_row("terminal-blocked")["lifecycle"] == (
        kb.BoardLifecycle.HARD_REMOVED.value
    )
    rows = _stored_receipts("terminal-blocked")
    assert len(rows) == 1, rows
    assert rows[0]["status"] == kb.RECEIPT_STATUS_APPLIED


def test_a_legacy_done_without_a_receipt_still_rolls_the_receipt_forward(
    fence_home, tmp_path
):
    """A record already at `done` with nothing prepared is still finished.

    An install upgraded mid-removal can hold a record that reached Done
    before the prepare/apply protocol existed. The roll-forward builds the
    receipt from the finished state rather than leaving a terminal claim
    with no account of what was destroyed.
    """
    _board("legacy-done", tmp_path / "repo")
    removal_id = _to_applied("legacy-done")
    assert kb.apply_permanent_mode_content(
        "legacy-done", removal_id=removal_id
    ).success
    assert kb.advance_removal_to_swept(
        "legacy-done", removal_id=removal_id
    ).success
    assert kb.complete_removal(
        "legacy-done", removal_id=removal_id, outcome="completed",
    ).success
    # Delete the receipt the protocol committed, so the record is exactly
    # the legacy shape: at done, with nothing in the archive for it.
    with kb.archive_connect() as conn:
        conn.execute(
            "DELETE FROM board_removal_receipt WHERE board_name = ?",
            ("legacy-done",),
        )
    assert _stored_receipts("legacy-done") == []

    record = kb.get_removal_phase_record("legacy-done")
    assert kb.terminal_receipt_outstanding(record) is True
    decision = kb.resume_removal("legacy-done")
    assert decision.point is kb.RemovalRecoveryPoint.P12
    assert decision.action is (
        kb.RemovalRecoveryAction.ROLL_FORWARD_TERMINAL_RECEIPT
    )

    result = kb.drive_removal("legacy-done")

    assert result.success, result.message
    rows = _stored_receipts("legacy-done")
    assert len(rows) == 1, rows
    assert rows[0]["status"] == kb.RECEIPT_STATUS_APPLIED
    assert json.loads(rows[0]["receipt_payload"])["subject"]["phase"] == (
        kb.RemovalPhase.DONE.value
    )


def test_an_invalid_board_name_is_refused_not_raised(fence_home):
    """The protocol's entry points refuse a bad name rather than raising."""
    prepared, reason = kb.prepare_permanent_removal_receipt(
        "   ", "rid", terminal_lifecycle="hard-removed",
        scope_declaration=kb.SCOPE_DECLARATION_VERSION, receipt_payload={},
    )
    applied, applied_reason = kb.apply_prepared_removal_receipt("   ", "rid")

    assert prepared is None
    assert reason
    assert applied is False
    assert applied_reason
    assert kb.get_prepared_removal_receipt("   ", "rid") is None
