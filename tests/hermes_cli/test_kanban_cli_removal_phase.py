"""``hermes kanban boards removal-phase`` (design revision 5, §6): the
read-only CLI surface for the recorded phase.

Wired the same way as ``backfill-fence``: a parser entry, a dispatch
branch, and a ``_cmd_...`` function. Read-only — it must never start,
advance, or abandon a removal.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_WORKTREE = Path(__file__).resolve().parents[2]
if str(_WORKTREE) not in sys.path:
    sys.path.insert(0, str(_WORKTREE))

from hermes_cli import kanban_db as kb
from tests.hermes_cli._kanban_fence_support import (
    apply_mode_specific_content,
    cli,
    create_fenced_board,
    start_removal,
)


def test_reports_no_removal_for_an_untouched_board(fence_home):
    create_fenced_board("cli-none")
    out = cli("boards removal-phase cli-none")
    assert "no removal recorded" in out


def test_json_reports_no_removal_as_a_null_removal_field(fence_home):
    create_fenced_board("cli-none-json")
    out = cli("boards removal-phase cli-none-json --json")
    payload = json.loads(out)
    assert payload["board"] == "cli-none-json"
    assert payload["removal"] is None


def test_reports_the_recorded_phase_after_intent(fence_home):
    create_fenced_board("cli-intent")
    result = start_removal("cli-intent", mode="permanent")
    assert result.success

    out = cli("boards removal-phase cli-intent")
    assert "permanent" in out
    assert "intent" in out
    assert result.removal_id in out


def test_json_output_carries_every_field(fence_home):
    create_fenced_board("cli-json")
    intent = start_removal("cli-json", mode="reversible")
    fenced = kb.advance_removal_to_fenced("cli-json", removal_id=intent.removal_id)
    assert fenced.success

    out = cli("boards removal-phase cli-json --json")
    payload = json.loads(out)
    assert payload["board"] == "cli-json"
    assert payload["removal_id"] == intent.removal_id
    assert payload["mode"] == "reversible"
    assert payload["phase"] == "fenced"
    assert payload["quiescence_deadline"] == fenced.record.quiescence_deadline
    assert payload["deadline_basis"] == fenced.record.deadline_basis
    assert payload["outcome"] is None


def test_the_command_is_read_only_it_never_advances_anything(fence_home):
    """Calling the CLI read repeatedly must never itself drive the phase
    forward — only the module-level advance functions may do that."""
    create_fenced_board("cli-readonly")
    intent = start_removal("cli-readonly", mode="reversible")

    for _ in range(5):
        cli("boards removal-phase cli-readonly")
        cli("boards removal-phase cli-readonly --json")

    record = kb.get_removal_phase_record("cli-readonly")
    assert record.phase == kb.RemovalPhase.INTENT


def test_missing_slug_is_a_usage_error():
    from hermes_cli.kanban import run_slash
    out = run_slash("boards removal-phase")
    assert "removal-phase" in out or "usage" in out.lower() or "error" in out.lower()


def test_a_recorded_operator_item_is_surfaced(fence_home):
    """IN-3: a resource the removal could not prove released is recorded
    with the exact resource and the reason — and SURFACED."""
    create_fenced_board("cli-operator-item")
    conn = kb.connect(board="cli-operator-item")
    task_id = kb.create_task(conn, title="work", assignee="worker")
    with kb.write_txn(conn):
        conn.execute(
            "INSERT INTO run_sandbox_cleanup_intents ("
            "  task_id, run_id, profile, generation, sandbox_id, "
            "  provision_event_id, attempt_count, next_attempt_at"
            ") VALUES (?, 1, 'worker', 1, 'sbx-cli-exact', 1, 0, 0)",
            (task_id,),
        )
    conn.close()

    intent = start_removal("cli-operator-item", mode="reversible")
    removal_id = intent.removal_id
    assert kb.advance_removal_to_fenced(
        "cli-operator-item", removal_id=removal_id
    ).success
    assert kb.advance_removal_to_quiesced(
        "cli-operator-item", removal_id=removal_id
    ).success
    assert kb.advance_removal_to_carried(
        "cli-operator-item", removal_id=removal_id
    ).success
    now = 5_000_000
    for _ in range(kb.RELEASE_MAX_ATTEMPTS):
        kb.advance_removal_to_released(
            "cli-operator-item", removal_id=removal_id, now=now
        )
        now += kb.RELEASE_MAX_WAIT_SECONDS

    payload = json.loads(cli("boards removal-phase cli-operator-item --json"))
    assert len(payload["operator_items"]) == 1
    assert payload["operator_items"][0]["identity"]["sandbox_id"] == "sbx-cli-exact"

    text = cli("boards removal-phase cli-operator-item")
    assert "operator item" in text
    assert "sbx-cli-exact" in text


def test_the_outstanding_mode_specific_application_is_surfaced(fence_home):
    """IN-3, §6.6: the declared exception that is keeping Done refused is
    reported by the read-only surface, in BOTH output modes."""
    create_fenced_board("cli-mode-content")
    removal_id = start_removal("cli-mode-content", mode="permanent").removal_id
    assert kb.advance_removal_to_fenced(
        "cli-mode-content", removal_id=removal_id
    ).success
    for driver in (
        kb.advance_removal_to_quiesced,
        kb.advance_removal_to_carried,
        kb.advance_removal_to_released,
        kb.advance_removal_to_applied,
        kb.advance_removal_to_swept,
    ):
        assert driver("cli-mode-content", removal_id=removal_id).success
    assert kb.complete_removal(
        "cli-mode-content", removal_id=removal_id, outcome="probe"
    ).success is False

    payload = json.loads(cli("boards removal-phase cli-mode-content --json"))
    assert payload["phase"] == "swept"
    assert payload["applied_mode_content"]["state"] == (
        kb.APPLIED_MODE_CONTENT_OUTSTANDING
    )
    assert payload["applied_mode_content"]["rule"] == "§7.2"
    items = [
        item for item in payload["operator_items"]
        if item["kind"] == kb.APPLIED_MODE_CONTENT_ITEM_KIND
    ]
    assert len(items) == 1
    assert items[0]["identity"]["step"] == "§7.2"

    text = cli("boards removal-phase cli-mode-content")
    assert "applied mode content: OUTSTANDING" in text
    assert "§7.2" in text
    assert "operator item" in text


def test_a_performed_mode_specific_application_is_surfaced_too(fence_home):
    create_fenced_board("cli-mode-done")
    removal_id = start_removal("cli-mode-done", mode="permanent").removal_id
    assert kb.advance_removal_to_fenced(
        "cli-mode-done", removal_id=removal_id
    ).success
    for driver in (
        kb.advance_removal_to_quiesced,
        kb.advance_removal_to_carried,
        kb.advance_removal_to_released,
        kb.advance_removal_to_applied,
        kb.advance_removal_to_swept,
    ):
        assert driver("cli-mode-done", removal_id=removal_id).success
    apply_mode_specific_content("cli-mode-done", removal_id)

    payload = json.loads(cli("boards removal-phase cli-mode-done --json"))
    assert payload["applied_mode_content"]["state"] == (
        kb.APPLIED_MODE_CONTENT_APPLIED
    )
    assert payload["applied_mode_content"]["evidence"]["checks"]
    assert payload["operator_items"] == []

    text = cli("boards removal-phase cli-mode-done")
    assert "applied mode content: performed" in text
    assert "operator item" not in text


def test_json_output_carries_the_recorded_scope_declaration_version(fence_home):
    create_fenced_board("cli-scope")
    start_removal("cli-scope", mode="reversible")

    payload = json.loads(cli("boards removal-phase cli-scope --json"))

    assert payload["scope_declaration_version"] == kb.SCOPE_DECLARATION_VERSION
    assert payload["operator_items"] == []
