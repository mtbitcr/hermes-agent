"""Classify approved plan changes through the daily review tick."""

import json
from datetime import datetime, timezone

import pytest
import yaml

from hermes_cli import kanban_db, owner_project_review as review, owner_workspace, projects_db
from hermes_cli import owner_project_review_record as records, owner_project_summary_events as events
from tests.hermes_cli.test_owner_project_review_plans import _approved_plan, _logical_state
from tests.hermes_cli.test_owner_project_review_record import _written_at, root  # noqa: F401
from tests.hermes_cli.test_owner_workspace import _bootstrap_board, ctx  # noqa: F401

NOW = int(datetime(2026, 11, 10, 7, tzinfo=timezone.utc).timestamp())


@pytest.mark.parametrize("action,fault", [
    ("replace", None), ("split", None), ("merge", None), ("cancel", None),
    ("replace", "comment"), ("replace", "reopen"),
    ("replace", "receipt"), ("replace", "event"),
    ("replace", "forged_change"), ("replace", "overflow"),
    ("replace", "fresh_reopen"),
])
def test_only_unchanged_approved_plan_changes_supersede_earlier_actionable_work(root, ctx, action, fault):
    earlier = NOW - 86400
    with _written_at(earlier):
        setup = _bootstrap_board(ctx)
        initial = _approved_plan(ctx, setup)
    [old] = initial["created_task_ids"]
    with projects_db.connect_closing(root / "projects.db") as registry:
        project = projects_db.get_project(registry, setup["project_id"])
    path = kanban_db.kanban_db_path(board=project.board_slug)
    with kanban_db.connect_closing(path, board=project.board_slug) as board:
        task = kanban_db.get_task(board, old)
        target = {"task_id": old, "expected_status": task.status,
                  "expected_revision": kanban_db.task_event_revision(board, old)}
        if action == "cancel":
            with _written_at(earlier):
                kanban_db.create_task(board, title="Check the arrival time", project_id=project.id)
        if action == "merge":
            with _written_at(earlier):
                sibling = kanban_db.create_task(board, title="Plan the menu", project_id=project.id)
            sibling_task = kanban_db.get_task(board, sibling)
            sibling_ref = {"task_id": sibling, "expected_status": sibling_task.status,
                           "expected_revision": kanban_db.task_event_revision(board, sibling)}
        cursor = board.execute("SELECT MAX(id) FROM task_events").fetchone()[0]
    with projects_db.connect_closing(root / "projects.db") as registry:
        projects_db.write_project_review_brief(
            registry, project.id, review_date=review.vienna_date(earlier), covered_from=None,
            covered_to=review.vienna_date(earlier), headline="The room needs work.",
            what_changed="Book the room is ready.", next_step="Book the room.",
            classification=[{"item": old, "class": "still actionable"}], written_at=earlier,
        )
        with kanban_db.write_txn(registry):
            registry.execute(
                "INSERT INTO project_review_state "
                "(project_id,event_cursor,last_reviewed_date,last_news_date,updated_at) VALUES (?,?,?,?,?)",
                (project.id, cursor, review.vienna_date(earlier), review.vienna_date(earlier), earlier),
            )
    replacement = {
        "title": "Confirm the guest list", "body": "Confirm the names before sending invitations.",
        "assignee": "default", "execution_tier": "routine", "risk_tier": 0,
    }
    change = {"action": action, "reason": "The venue now supplies the room.", "target": target}
    if action == "replace":
        change["replacement"] = replacement
    elif action == "split":
        change["replacements"] = [
            {**replacement, "parents": []},
            {**replacement, "title": "Check dietary needs", "parents": [0]},
        ]
    elif action == "merge":
        del change["target"]
        change.update(targets=[target, sibling_ref], replacement=replacement)
    with _written_at(NOW):
        _approved_plan(ctx, setup, key="change-room-work", changes=[change])
        if fault == "fresh_reopen":
            with kanban_db.connect_closing(path, board=project.board_slug) as board:
                assert kanban_db.cas_transition_task(
                    board, old, expected_status="archived", expected_revision=kanban_db.task_event_revision(board, old),
                    to_status="ready", event_kind="owner_move",
                    event_payload={"actor": "default", "reason": "The owner needs the room work again."},
                )["moved"]
    frozen = records.build_review_input(project, events.Cursors(cursor, 0, 0), earlier)
    [plan] = [item for item in frozen["sources"]["proposals"] if item["at"] == NOW]
    (root / "config.yaml").write_text(yaml.safe_dump({"owner_review": {"enabled": True, "allowlist": [project.id]}}))
    assert owner_workspace.project_steward_snapshot(project_id=project.id)["execution"]["state"] not in {"deleted", "complete"}

    assert review.review_tick(NOW) == 1
    with projects_db.connect_closing(root / "projects.db") as registry:
        review_task = registry.execute(
            "SELECT open_task_id FROM project_review_state WHERE project_id=?", (project.id,),
        ).fetchone()[0]
    payload = {
        "summary": {"outcome": "The venue supplies the room.", "next_action": "Confirm the guest list.",
                    "owner_decision": None},
        "brief": {"headline": "The venue supplies the room.", "what_changed": "Book the room has been replaced.",
                  "next_step": "Confirm the guest list.", "needs_you": None},
        "classification": [{"item": old, "class": "current", "evidence": []}], "proposed_closures": [],
    }
    if fault == "fresh_reopen":
        payload["classification"] = [{"item": old, "class": "still actionable", "evidence": [f"tasks:{old}"]}]
        payload["summary"].update(outcome="The owner reopened the room work.", next_action="Book the room.")
        payload["brief"].update(what_changed="Book the room is ready again.", next_step="Book the room.")
    with kanban_db.connect_closing(kanban_db.kanban_db_path(board="project-summaries"), board="project-summaries") as board:
        assert kanban_db.claim_task(board, review_task) is not None
        assert kanban_db.complete_task(board, review_task, summary="Reviewed the changed plan.",
                                       metadata={"project_review": payload})
    if fault in {"comment", "reopen", "event"}:
        with kanban_db.connect_closing(path, board=project.board_slug) as board:
            if fault == "comment":
                kanban_db.add_comment(board, old, "default", "Check the room work again.")
            elif fault == "reopen":
                assert kanban_db.cas_transition_task(
                    board, old, expected_status="archived", expected_revision=kanban_db.task_event_revision(board, old),
                    to_status="ready", event_kind="owner_move",
                )["moved"]
            else:
                with kanban_db.write_txn(board):
                    board.execute("DELETE FROM task_events WHERE task_id=? AND kind='owner_project_plan_change'", (old,))
    elif fault == "receipt":
        with projects_db.connect_closing(root / "projects.db") as registry, kanban_db.write_txn(registry):
            registry.execute(
                "DELETE FROM owner_workspace_receipts WHERE operation='owner_project_plan_commit' AND idempotency_key=?",
                ("change-room-work",),
            )
    elif fault in {"forged_change", "overflow"}:
        with kanban_db.connect_closing(kanban_db.kanban_db_path(board="project-summaries"), board="project-summaries") as board:
            task = kanban_db.get_task(board, review_task)
            lines = task.body.splitlines()
            changed = False
            for index, line in enumerate(lines):
                if not line.startswith("| "):
                    continue
                item = json.loads(line[2:])
                if item.get("id") != plan["id"]:
                    continue
                if fault == "overflow":
                    item["at"] = 2**64
                else:
                    item["changes"][0]["action"] = "cancel"
                lines[index] = "| " + json.dumps(item, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
                changed = True
            assert changed
            with kanban_db.write_txn(board):
                board.execute("UPDATE tasks SET body=? WHERE id=?", ("\n".join(lines), review_task))
    before = _logical_state(path)

    assert review.review_tick(NOW + 1) == 0

    with projects_db.connect_closing(root / "projects.db") as registry:
        brief = projects_db.get_project_review_brief(registry, project.id)
    if fault not in {None, "fresh_reopen"}:
        assert brief["review_date"] == review.vienna_date(earlier)
        assert _logical_state(path) == before
        return
    assert brief["review_date"] == review.vienna_date(NOW)
    [classified] = records.build_review_input(project, events.Cursors(), None)["sources"]["earlier_reviews"][0]["classification"]
    assert classified["item"] == old
    if fault == "fresh_reopen":
        assert classified["class"] == "still actionable"
        assert classified["evidence"] == [f"tasks:{old}"]
        assert classified["actor"] == "worker"
    else:
        assert classified["class"] == "superseded"
        assert classified["evidence"] == [f"proposals:{plan['id']}"]
    assert _logical_state(path) == before
