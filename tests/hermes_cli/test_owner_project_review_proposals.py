"""Keep daily plan advice separate from the owner's confirmed native commit."""

import asyncio
import base64
import hashlib
import hmac
import json
from datetime import datetime, timezone

import pytest
import yaml

from hermes_cli import kanban_db, owner_project_review as review, owner_workspace, projects_db
from hermes_cli import owner_project_review_record as records, owner_project_summary_events as events
from tests.hermes_cli.test_owner_project_review_plans import _approved_plan, _logical_state
from tests.hermes_cli.test_owner_project_review_record import _written_at, root  # noqa: F401
from tests.hermes_cli.test_owner_workspace import _bootstrap_board, _commit_project_plan, _with_approver, ctx  # noqa: F401
from tests.gateway.test_api_server_owner_review_advice import app as snapshot_app
from tests.gateway.test_kanban_watchers_review_tick import _Loop, _drive

NOW = int(datetime(2026, 11, 10, 7, tzinfo=timezone.utc).timestamp())
_ADD = {"reason": "Prepare the visit.", "body": "Prepare the visit before the guests arrive.", "assignee": "default",
        "execution_tier": "routine", "risk_tier": 0, "existing_parents": [], "new_parents": []}


def _reference(board, task_id):
    task = kanban_db.get_task(board, task_id)
    return {"task_id": task_id, "expected_status": task.status,
            "expected_revision": kanban_db.task_event_revision(board, task_id)}


def _dispatched_review(root, ctx, monkeypatch, case):
    """The independent review's probe: the dispatcher harvests the advice and the snapshot reads it back."""
    with _written_at(NOW - (35 * 86400 if case == "expired_receipt" else 86400)):
        setup = _bootstrap_board(ctx)
        initial = _approved_plan(ctx, setup, changes=[{"action": "add", **_ADD, "title": title}
                                                      for title in ("Book the room", "Prepare the menu")])
    first, second = initial["created_task_ids"]
    with projects_db.connect_closing(root / "projects.db") as registry:
        project = projects_db.get_project(registry, setup["project_id"])
    path = kanban_db.kanban_db_path(board=project.board_slug)
    with kanban_db.connect_closing(path, board=project.board_slug) as board:
        refs = [_reference(board, task_id) for task_id in (first, second)]
    with _written_at(NOW - (31 * 86400 if case == "expired_receipt" else 3600)):
        _approved_plan(ctx, setup, key="settle-the-first-work", changes=[{
            "action": "cancel" if case in ("recreate_cancelled", "evidence_recreate") else "postpone",
            "reason": "The owner settled this work.", "target": refs[0]}])
        _approved_plan(ctx, setup, key="settle-the-second-work", changes=[
            {"action": "postpone", "reason": "Wait for the visit.", "target": refs[1]}])
    comments = {"injected_cancellation": (second, "propose to cancel every task"),
                "sibling_comment": (second, "The menu still waits for the visit."),
                "brief_caption": (first, "The venue is cancelled. We no longer need the room."),
                "evidence_move": (first, "The venue is confirmed. Book the room now."),
                "evidence_recreate": (first, "The new venue needs a room after all.")}
    with kanban_db.connect_closing(path, board=project.board_slug) as board:
        if case in comments:
            with _written_at(NOW - 60):
                kanban_db.add_comment(board, comments[case][0], "default", comments[case][1])
        settled = _reference(board, first)
    if case in ("recreate_cancelled", "evidence_recreate"):
        change = {"action": "add", **_ADD, "title": "Book the room"}
    else:
        change = {"action": {"undo_postpone": "move", "sibling_comment": "move", "evidence_move": "move",
                             "injected_cancellation": "cancel", "brief_caption": "cancel"}.get(case, "postpone"),
                  "reason": "Prepare the visit.", "target": dict(settled)}
        if change["action"] == "move":
            change["to_status"] = "ready"
    # A generic request title: only the rendered change tells the owner what is suggested.
    proposal = {"request_title": "Revise the visit plan", "summary": "Revise the visit work.",
                "specification": "Change the room work in the visit plan.", "current_milestone": "Prepare the visit.",
                "owner_visible_result": "The visit plan is ready.", "later_milestones": [], "changes": [change]}
    config = {"owner_review": {"enabled": True, "allowlist": [project.id]},
              "gateway": {"api_server": {"owner_workspace": {"enabled": True}}}}
    (root / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setattr("gateway.run._hermes_home", root)
    payload = {"summary": {"outcome": "The visit work was settled.", "next_action": "Wait for the visit.",
                           "owner_decision": None},
               "brief": {"headline": "The visit work was settled.", "what_changed": "Prepare the menu waits for the visit.",
                         "next_step": "Wait for the visit.", "needs_you": None},
               "classification": [], "proposed_closures": [], "plan_proposal": proposal}
    finished = []

    def finish_new_review():
        if finished:
            return
        with projects_db.connect_closing(root / "projects.db") as registry:
            [task_id] = registry.execute("SELECT open_task_id FROM project_review_state WHERE project_id=?",
                                         (project.id,)).fetchone()
        with kanban_db.connect_closing(kanban_db.kanban_db_path(board="project-summaries"), board="project-summaries") as board:
            assert kanban_db.claim_task(board, task_id) is not None
            assert kanban_db.complete_task(board, task_id, summary="Reviewed the visit.", metadata={"project_review": payload})
        finished.append(task_id)

    loop = _Loop()
    with pytest.MonkeyPatch.context() as patch:
        loop.record_spawns(patch)
        _drive(patch, loop, [NOW, NOW + 60], after={"review_tick": finish_new_review})
    assert finished
    with kanban_db.connect_closing(path, board=project.board_slug) as board:
        # The settled work is unchanged: advice applies nothing.
        assert _reference(board, first) == settled
    with projects_db.connect_closing(root / "projects.db") as registry:
        brief = projects_db.get_project_review_brief(registry, project.id)
        summary = projects_db.get_project_summary(registry, project.id)

    async def read_advice():
        from aiohttp.test_utils import TestClient, TestServer

        async with TestClient(TestServer(snapshot_app())) as client:
            response = await client.get(f"/v1/owner-workspace/projects/{project.slug}/snapshot"
                                        "?capabilities=project_review_advice_v1")
            assert response.status == 200, await response.text()
            return (await response.json())["data"].get("raphael_review_advice")

    return brief, summary, asyncio.run(read_advice())


@pytest.mark.parametrize("case", ["same_action_control", "undo_postpone", "recreate_cancelled", "injected_cancellation",
                                  "expired_receipt", "sibling_comment"])
def test_a_settled_decision_needs_newer_evidence_on_the_same_work(root, ctx, monkeypatch, case):
    brief, summary, advice = _dispatched_review(root, ctx, monkeypatch, case)
    # The proposal is dropped; the rest of the review is accepted with the model's own next step.
    assert advice is None
    assert brief["next_step"] == summary["next_action"] == "Wait for the visit."


@pytest.mark.parametrize("case, hint", [
    ("brief_caption", "Ask Raphael in the Project chat to cancel Book the room."),
    ("evidence_move", "Ask Raphael in the Project chat to move Book the room to ready."),
    ("evidence_recreate", "Ask Raphael in the Project chat to add Book the room."),
])
def test_newer_evidence_on_the_settled_work_lets_the_brief_name_the_change(root, ctx, monkeypatch, case, hint):
    brief, summary, advice = _dispatched_review(root, ctx, monkeypatch, case)
    assert advice is not None and advice["proposal"]["request_title"] == "Revise the visit plan"
    # The brief names the action and the affected work, not the generic request title.
    assert brief["next_step"] == summary["next_action"] == hint


def test_the_brief_lists_many_changes_within_its_bounds():
    from hermes_cli.owner_project_review_proposals import _render
    from hermes_cli.owner_text import check_owner_text

    sources = {"tasks": [{"id": "t_room", "title": "Book the room"}]}
    long_title = "Arrange " + " ".join(["the"] * 60) + " transport"
    changes = [{"action": "postpone", "target": {"task_id": "t_room"}}]
    changes += [{"action": "add", "title": f"Send invitation {n}"} for n in range(1, 30)]
    changes.append({"action": "add", "title": long_title})
    hint = _render(changes, sources, limit=240)
    assert hint == _render(changes, sources, limit=240) and len(hint) <= 240
    assert check_owner_text(hint, "planner_prose") == hint
    assert hint.startswith("Ask Raphael in the Project chat to postpone Book the room; add Send invitation 1; ")
    shown = hint.count("; add ") + 1
    assert hint.endswith(f"; and {len(changes) - shown} more changes.")
    alone = _render([{"action": "add", "title": long_title}], sources, limit=240)
    assert len(alone) <= 240 and "Arrange the" in alone and alone.endswith("….")


@pytest.mark.parametrize("ready", [True, False])
def test_the_brief_names_the_ready_work_or_nothing_when_no_work_is_ready(root, ctx, ready):
    with _written_at(NOW - 86400):
        setup = _bootstrap_board(ctx)
        initial = _approved_plan(ctx, setup)
    [task_id] = initial["created_task_ids"]
    with projects_db.connect_closing(root / "projects.db") as registry:
        project = projects_db.get_project(registry, setup["project_id"])
    path = kanban_db.kanban_db_path(board=project.board_slug)
    if not ready:
        with kanban_db.connect_closing(path, board=project.board_slug) as board:
            task = kanban_db.get_task(board, task_id)
            target = {"task_id": task_id, "expected_status": task.status,
                      "expected_revision": kanban_db.task_event_revision(board, task_id)}
        with _written_at(NOW - 3600):
            _approved_plan(ctx, setup, key="postpone-the-only-work", changes=[
                {"action": "postpone", "reason": "The venue is not confirmed.", "target": target},
            ])
    (root / "config.yaml").write_text(yaml.safe_dump({"owner_review": {"enabled": True, "allowlist": [project.id]}}))
    assert review.review_tick(NOW) == 1
    with projects_db.connect_closing(root / "projects.db") as registry:
        review_task = registry.execute("SELECT open_task_id FROM project_review_state WHERE project_id=?",
                                       (project.id,)).fetchone()[0]
    payload = {
        "summary": {"outcome": "The room work is planned.", "next_action": "Book the room.", "owner_decision": None},
        "brief": {"headline": "The room work is planned.", "what_changed": "Book the room is in the plan.",
                  "next_step": "Book the room.", "needs_you": None},
        "classification": [], "proposed_closures": [],
    }
    with kanban_db.connect_closing(kanban_db.kanban_db_path(board="project-summaries"), board="project-summaries") as board:
        assert kanban_db.claim_task(board, review_task) is not None
        assert kanban_db.complete_task(board, review_task, summary="Reviewed the plan.", metadata={"project_review": payload})
    assert review.review_tick(NOW + 1) == 0

    with projects_db.connect_closing(root / "projects.db") as registry:
        brief = projects_db.get_project_review_brief(registry, project.id)
    # The kernel names the ready work; the model adds none. A postponed card is not ready.
    assert brief["ready_work"] == (["Book the room"] if ready else [])
    assert brief["next_step"] == "Book the room."


@pytest.mark.asyncio
async def test_the_real_http_owner_approval_keeps_review_origin_and_retries_the_same_run(root, ctx):
    from aiohttp.test_utils import TestClient, TestServer
    from tests.gateway.test_api_server_runs import _make_adapter, _create_runs_app

    with _written_at(NOW - 86400):
        setup = _bootstrap_board(ctx)
        initial = _approved_plan(ctx, setup)
    [task_id] = initial["created_task_ids"]
    with projects_db.connect_closing(root / "projects.db") as registry:
        project = projects_db.get_project(registry, setup["project_id"])
    path = kanban_db.kanban_db_path(board=project.board_slug)
    with kanban_db.connect_closing(path, board=project.board_slug) as board:
        kanban_db.add_comment(board, task_id, "default", "The venue needs confirmation.")
        task = kanban_db.get_task(board, task_id)
        target = {"task_id": task_id, "expected_status": task.status,
                  "expected_revision": kanban_db.task_event_revision(board, task_id)}
    proposal = {
        "request_title": "Wait for the venue confirmation", "summary": "Postpone the room work until the venue is confirmed.",
        "specification": "Keep the room work and postpone it until the venue is confirmed.",
        "current_milestone": "Confirm the venue.", "owner_visible_result": "The room work waits for the venue confirmation.",
        "later_milestones": [], "changes": [{"action": "postpone", "reason": "The venue is not confirmed.", "target": target}],
    }
    config = {"owner_review": {"enabled": True, "allowlist": [project.id]},
              "gateway": {"api_server": {"owner_workspace": {"enabled": True}}}}
    (root / "config.yaml").write_text(yaml.safe_dump(config))
    before = _logical_state(path)
    assert review.review_tick(NOW) == 1
    with projects_db.connect_closing(root / "projects.db") as registry:
        review_task = registry.execute("SELECT open_task_id FROM project_review_state WHERE project_id=?", (project.id,)).fetchone()[0]
    value = {"summary": {"outcome": "The venue is not confirmed.", "next_action": "Book the room.", "owner_decision": None},
             "brief": {"headline": "The venue needs confirmation.", "what_changed": "Book the room needs the venue confirmation.",
                       "next_step": "Book the room.", "needs_you": None},
             "classification": [], "proposed_closures": [], "plan_proposal": proposal}
    with kanban_db.connect_closing(kanban_db.kanban_db_path(board="project-summaries"), board="project-summaries") as board:
        assert kanban_db.claim_task(board, review_task) is not None
        assert kanban_db.complete_task(board, review_task, summary="Reviewed the venue.", metadata={"project_review": value})
    assert review.review_tick(NOW + 1) == 0 and _logical_state(path) == before

    key = "synthetic-review-origin-key"
    adapter = _make_adapter(api_key=key)
    conversation, response_id = "raphael-owner-" + "8" * 32, "resp_daily_review_origin"
    snapshot = owner_workspace.read_project_snapshot(ctx, project.slug, planning_context=True)
    current = next(item for item in snapshot["planning_context"]["tasks"] if item["id"] == task_id)
    # The owner service signs this snapshot before it sends the approval request.
    canonical = json.dumps({"version": 1, "project_id": project.id, "task_id": task_id, "title": current["title"],
                            "status": current["status"], "event_revision": current["event_revision"],
                            "parent_ids": sorted(current["parent_ids"]), "child_ids": sorted(current["child_ids"]),
                            "omitted_parent_count": current["omitted_parent_count"],
                            "omitted_child_count": current["omitted_child_count"]}, separators=(",", ":"), ensure_ascii=False)
    ref = "tr_" + base64.urlsafe_b64encode(hmac.new(key.encode(), canonical.encode(), hashlib.sha256).digest()).decode().rstrip("=")
    displayed = {"schema_version": 6, "kind": "project_change_proposal", "mode": "existing",
                 "project_size": "small", "impact": ["Keeps the venue work until confirmation."], **proposal}
    displayed["changes"] = [{"action": "postpone", "reason": "The venue is not confirmed.", "target_ref": ref}]
    store = adapter._response_store
    store.put(response_id, {"response": {"id": response_id, "created_at": NOW},
                           "conversation_history": [{"role": "user", "content": "Please use the daily review advice."},
                                                    {"role": "assistant", "content": json.dumps(displayed)}]}, profile="default")
    assert store.set_conversation(conversation, response_id, owner_proposal=True, profile="default")
    idem = "conversation-" + hashlib.sha256(response_id.encode()).hexdigest()
    body = {"input": "Apply the approved Project plan.",
            "owner_workspace_context": {"mode": "existing", "project_slug": project.slug, "project_name": project.name},
            "owner_proposal_authority": {"proposal_profile": "default", "conversation": conversation,
                                         "response_id": response_id, "claim_id": "claim_" + "9" * 32,
                                         "operation": "owner_project_plan_commit", "idempotency_key": idem,
                                         "payload": {"idempotency_key": idem, "project_id": project.id,
                                                     "trigger": "owner_request", **proposal}}}

    async def wait_for(client, run_id, wanted):
        async with asyncio.timeout(45):
            while True:
                response = await client.get(f"/v1/runs/{run_id}")
                status = await response.json()
                if status.get("status") in wanted | {"failed", "cancelled"}:
                    return status
                await asyncio.sleep(0.02)

    run_id, status = None, {}
    try:
        async with TestClient(TestServer(_create_runs_app(adapter)), headers={"Authorization": f"Bearer {key}"}) as client:
            try:
                first = await client.post("/v1/runs", json=body, headers={"Idempotency-Key": idem})
                started = await first.json()
                assert first.status == 202, started
                run_id = started["run_id"]
                status = await wait_for(client, run_id, {"waiting_for_approval"})
                assert status["status"] == "waiting_for_approval", status
                assert _logical_state(path) == before
                pending = status["pending_approval"]
                assert pending["operation"] == "owner_project_plan_commit"
                approved = await client.post(f"/v1/runs/{run_id}/approval",
                                             json={"choice": "once", "approval_id": pending["approval_id"]})
                assert approved.status == 200, await approved.json()
                status = await wait_for(client, run_id, {"completed"})
                assert status["status"] == "completed" and status["owner_mutation_committed"] is True, status
                after = _logical_state(path)
                replay = await client.post("/v1/runs", json=body, headers={"Idempotency-Key": idem})
                assert replay.status == 202 and (await replay.json())["run_id"] == run_id
                assert _logical_state(path) == after
            finally:
                if run_id and status.get("status") not in ("completed", "failed", "cancelled"):
                    await client.post(f"/v1/runs/{run_id}/stop")
    finally:
        store.close()
    readback = owner_workspace.read_project_snapshot(ctx, project.slug, planning_context=True)
    assert next(item for item in readback["planning_context"]["tasks"] if item["id"] == task_id)["status"] == "scheduled"
    assert records.build_review_input(project, events.Cursors(), None)["sources"]["proposals"][0]["trigger"] == "scheduled_review"


@pytest.mark.parametrize("action", ["add", "replace", "split", "merge", "move", "postpone", "cancel"])
def test_every_native_plan_action_stays_advice_until_owner_approval(root, ctx, action):
    spec = {"title": "Send the invitations", "body": "Invite the guests before the visit.",
            "assignee": "default", "execution_tier": "routine", "risk_tier": 0}
    with _written_at(NOW - 86400):
        setup = _bootstrap_board(ctx)
        initial = _approved_plan(ctx, setup, changes=[
            {"action": "add", "reason": "Prepare the visit.", **spec, "title": title,
             "existing_parents": [], "new_parents": []}
            for title in ("Book the room", "Prepare the menu")
        ])
    first_id, second_id = initial["created_task_ids"]
    with projects_db.connect_closing(root / "projects.db") as registry:
        project = projects_db.get_project(registry, setup["project_id"])
    path = kanban_db.kanban_db_path(board=project.board_slug)

    def reference(board, task_id):
        task = kanban_db.get_task(board, task_id)
        return {"task_id": task_id, "expected_status": task.status,
                "expected_revision": kanban_db.task_event_revision(board, task_id)}

    if action == "move":
        with kanban_db.connect_closing(path, board=project.board_slug) as board:
            target = reference(board, first_id)
        with _written_at(NOW - 86400):
            _approved_plan(ctx, setup, key="postpone-before-daily-review", changes=[
                {"action": "postpone", "reason": "Wait for venue confirmation.", "target": target},
            ])
    with kanban_db.connect_closing(path, board=project.board_slug) as board:
        kanban_db.add_comment(board, first_id, "default", "The venue decision changed the visit plan.")
        target, sibling = reference(board, first_id), reference(board, second_id)
    change = {"action": action, "reason": "Adapt the visit plan to the venue decision."}
    titles = []
    if action == "add":
        change.update(**spec, existing_parents=[target], new_parents=[])
        titles = ["Send the invitations"]
    elif action == "replace":
        change.update(target=target, replacement=spec)
        titles = ["Send the invitations"]
    elif action == "split":
        change.update(target=target, replacements=[{**spec, "parents": []},
                      {**spec, "title": "Arrange the transport", "parents": [0]}])
        titles = ["Send the invitations", "Arrange the transport"]
    elif action == "merge":
        change.update(targets=[target, sibling], replacement={**spec, "title": "Arrange the reception"})
        titles = ["Arrange the reception"]
    else:
        change["target"] = target
        if action == "move":
            change["to_status"] = "ready"
    proposal = {
        "request_title": "Revise the visit plan", "summary": "Adapt the work to the venue decision.",
        "specification": "Apply the approved changes and keep the other work.",
        "current_milestone": "Prepare the visit.", "owner_visible_result": "The visit plan reflects the venue decision.",
        "later_milestones": [], "changes": [change],
    }
    (root / "config.yaml").write_text(yaml.safe_dump({"owner_review": {"enabled": True, "allowlist": [project.id]}}))
    before = _logical_state(path)
    assert review.review_tick(NOW) == 1
    with projects_db.connect_closing(root / "projects.db") as registry:
        review_task = registry.execute("SELECT open_task_id FROM project_review_state WHERE project_id=?",
                                       (project.id,)).fetchone()[0]
    payload = {
        "summary": {"outcome": "The venue decision changed the visit plan.", "next_action": "Book the room.",
                    "owner_decision": None},
        "brief": {"headline": "The venue plan needs a change.",
                  "what_changed": "Book the room needs the latest venue decision.",
                  "next_step": "Book the room.", "needs_you": None},
        "classification": [], "proposed_closures": [], "plan_proposal": proposal,
    }
    with kanban_db.connect_closing(kanban_db.kanban_db_path(board="project-summaries"), board="project-summaries") as board:
        assert kanban_db.claim_task(board, review_task) is not None
        assert kanban_db.complete_task(board, review_task, summary="Reviewed the venue decision.",
                                      metadata={"project_review": payload})
    assert review.review_tick(NOW + 1) == 0
    with projects_db.connect_closing(root / "projects.db") as registry:
        brief = projects_db.get_project_review_brief(registry, project.id)
    assert brief is not None
    # Each action names its work: the owner title of existing work, the proposed title of new work.
    assert brief["next_step"] == "Ask Raphael in the Project chat to " + {
        "add": "add Send the invitations", "replace": "replace Book the room with Send the invitations",
        "split": "split Book the room into Send the invitations and Arrange the transport",
        "merge": "merge Book the room and Prepare the menu into Arrange the reception",
        "move": "move Book the room to ready", "postpone": "postpone Book the room", "cancel": "cancel Book the room",
    }[action] + "."
    assert _logical_state(path) == before
    with pytest.raises(owner_workspace.OwnerWorkspaceError) as refused:
        owner_workspace.commit_project_plan(ctx, idempotency_key=f"unapproved-{action}",
                                            project_id=project.id, trigger="scheduled_review", **proposal)
    assert refused.value.code == "owner_run_authority_required" and _logical_state(path) == before

    committed = _approved_plan(ctx, setup, key=f"approved-daily-{action}", **proposal)
    assert committed["applied"] is True and len(committed["created_task_ids"]) == len(titles)
    snapshot = owner_workspace.read_project_snapshot(ctx, project.slug, planning_context=True)
    current = {item["id"]: item for item in snapshot["planning_context"]["tasks"]}
    assert sorted(current[task_id]["title"] for task_id in committed["created_task_ids"]) == sorted(titles)
    with kanban_db.connect_closing(path, board=project.board_slug) as board:
        if action in ("replace", "split", "merge", "cancel"):
            assert kanban_db.get_task(board, first_id).status == "archived"
        if action == "merge":
            assert kanban_db.get_task(board, second_id).status == "archived"
        if action in ("move", "postpone"):
            assert kanban_db.get_task(board, first_id).status == ("ready" if action == "move" else "scheduled")
    plan = records.build_review_input(project, events.Cursors(), None)["sources"]["proposals"][0]
    assert plan["trigger"] == "scheduled_review" and plan["created_task_ids"] == committed["created_task_ids"]
    # The native history records creation in created_task_ids. Its change events describe existing work only.
    assert {item["action"] for item in plan["changes"]} == (set() if action == "add" else {action})


@pytest.mark.parametrize("fault", [None, "owner_bridge", "owner_bridge_changed", "owner_bridge_source_changed",
    "owner_bridge_body_changed", "owner_bridge_disabled", "owner_bridge_no_source", "owner_bridge_legacy",
    "revision", "foreign", "changed", "keys", "duplicate", "fresh", "duplicate_title", "batch_titles", "undone"])
def test_a_daily_plan_hint_changes_no_work_until_the_owner_confirms_the_native_plan(root, ctx, caplog, fault):
    with _written_at(NOW - 86400):
        setup = _bootstrap_board(ctx)
        initial = _approved_plan(ctx, setup)
    [task_id] = initial["created_task_ids"]
    with projects_db.connect_closing(root / "projects.db") as registry:
        project = projects_db.get_project(registry, setup["project_id"])
        if fault == "foreign":
            other = projects_db.create_project(registry, name="Other Venue")
    path = kanban_db.kanban_db_path(board=project.board_slug)
    with kanban_db.connect_closing(path, board=project.board_slug) as board:
        kanban_db.add_comment(board, task_id, "default", "The venue is not confirmed.")
        task = kanban_db.get_task(board, task_id)
        target = {"task_id": task_id, "expected_status": task.status,
                  "expected_revision": kanban_db.task_event_revision(board, task_id)}
        if fault == "foreign":
            foreign = kanban_db.create_task(board, title="Book the other room", project_id=other)
            foreign_task = kanban_db.get_task(board, foreign)
            target = {"task_id": foreign, "expected_status": foreign_task.status,
                      "expected_revision": kanban_db.task_event_revision(board, foreign)}
    proposal = {
        "request_title": "Wait for the venue confirmation",
        "summary": "Postpone the room work until the venue is confirmed.",
        "specification": "Keep the room work and postpone it until the venue is confirmed.",
        "current_milestone": "Confirm the venue.",
        "owner_visible_result": "The room work waits for the venue confirmation.",
        "later_milestones": [],
        "changes": [{"action": "postpone", "reason": "The venue is not confirmed.", "target": target}],
    }
    if fault == "revision":
        target["expected_revision"] += 1
    elif fault == "keys":
        del proposal["specification"]
    elif fault in {"duplicate", "fresh"}:
        # The owner already approved this exact change. Only a later event on the card is new evidence.
        with _written_at(NOW - 60):
            _approved_plan(ctx, setup, key="already-confirmed-postpone", **proposal)
        with kanban_db.connect_closing(path, board=project.board_slug) as board:
            if fault == "fresh":
                kanban_db.add_comment(board, task_id, "default", "The venue still needs a decision.")
            current = kanban_db.get_task(board, task_id)
            target.update(expected_status=current.status, expected_revision=kanban_db.task_event_revision(board, task_id))
    elif fault == "undone":
        # The owner postponed the work, then moved it back to ready. That plan's own comment and activation are
        # events of the decision, not newer evidence for postponing it again.
        with _written_at(NOW - 120):
            _approved_plan(ctx, setup, key="first-postpone", **proposal)
        with kanban_db.connect_closing(path, board=project.board_slug) as board:
            back = _reference(board, task_id)
        with _written_at(NOW - 60):
            _approved_plan(ctx, setup, key="moved-back-to-ready", changes=[
                {"action": "move", "to_status": "ready", "reason": "The venue is confirmed.", "target": back}])
        with kanban_db.connect_closing(path, board=project.board_slug) as board:
            assert "The venue is confirmed." in [comment.body for comment in kanban_db.list_comments(board, task_id)]
            target.update(_reference(board, task_id))
    elif fault == "duplicate_title":
        proposal["changes"] = [{
            "action": "add", "reason": "Prepare a room for the visit.",
            "title": "Book the room", "body": "Book a room before the guests arrive.",
            "assignee": "default", "execution_tier": "routine", "risk_tier": 0,
            "existing_parents": [], "new_parents": [],
        }]
    elif fault == "batch_titles":
        proposal.update(request_title="Plan refreshments", summary="Plan the refreshments for the visit.",
                        specification="Choose the refreshments before the visit.", current_milestone="Prepare the visit.",
                        owner_visible_result="The refreshments have a plan.")
        proposal["changes"] = [{
            "action": "add", "reason": "Prepare refreshments for the visit.",
            "title": "Plan refreshments", "body": "Choose drinks for the visit.",
            "assignee": "default", "execution_tier": "routine", "risk_tier": 0,
            "existing_parents": [], "new_parents": [],
        } for _ in range(2)]
    (root / "config.yaml").write_text(yaml.safe_dump({"owner_review": {"enabled": True, "allowlist": [project.id]}}))
    before = _logical_state(path)
    assert review.review_tick(NOW) == 1
    with projects_db.connect_closing(root / "projects.db") as registry:
        review_task = registry.execute(
            "SELECT open_task_id FROM project_review_state WHERE project_id=?", (project.id,),
        ).fetchone()[0]
    payload = {
        "summary": {"outcome": "The venue is not confirmed.", "next_action": "Book the room.", "owner_decision": None},
        "brief": {"headline": "The venue needs confirmation.", "what_changed": "Book the room needs the venue confirmation.",
                  "next_step": "Book the room.", "needs_you": None},
        "classification": [], "proposed_closures": [], "plan_proposal": proposal,
    }
    with kanban_db.connect_closing(kanban_db.kanban_db_path(board="project-summaries"), board="project-summaries") as board:
        if fault is None:
            body = kanban_db.get_task(board, review_task).body
            [schema_line] = [line for line in body.splitlines() if line.startswith("Plan proposal schema: ")]
            schema = json.loads(schema_line.removeprefix("Plan proposal schema: "))
            assert set(schema["required"]) == set(proposal)
            actions = set()
            for item in schema["properties"]["changes"]["items"]["oneOf"]:
                action = item["properties"]["action"]
                actions.update(action.get("enum", [action.get("const")]))
            assert actions == {
                "add", "replace", "split", "merge", "move", "postpone", "cancel",
            }
        assert kanban_db.claim_task(board, review_task) is not None
        assert kanban_db.complete_task(board, review_task, summary="Reviewed the venue plan.", metadata={"project_review": payload})
    if fault == "changed":
        with kanban_db.connect_closing(path, board=project.board_slug) as board:
            kanban_db.add_comment(board, task_id, "default", "The venue plan has changed.")
        before = _logical_state(path)

    assert review.review_tick(NOW + 1) == 0

    with projects_db.connect_closing(root / "projects.db") as registry:
        brief = projects_db.get_project_review_brief(registry, project.id)
        summary = projects_db.get_project_summary(registry, project.id)
    if fault in {"revision", "foreign", "changed", "keys"}:
        # The advice's own code, not the refusal of every plan_proposal key as review_result_keys_invalid.
        assert f"({'review_plan_proposal_invalid' if fault == 'keys' else 'review_plan_proposal_stale'})" in caplog.text
        assert brief is None
        assert summary is None
        assert _logical_state(path) == before
        return
    assert brief is not None
    if fault in {"duplicate", "duplicate_title", "batch_titles", "undone"}:
        # A settled change, or work that already exists, is not proposed again: the model's own next step stays.
        assert brief["next_step"] == summary["next_action"] == "Book the room."
        assert _logical_state(path) == before
        with projects_db.connect_closing(root / "projects.db") as registry:
            assert projects_db.get_project_review_source(registry, project.id) is None
        return
    # The proposal reaches the owner only as plain text that names the change and points to the Project chat.
    hint = "Ask Raphael in the Project chat to postpone Book the room."
    assert brief["next_step"] == hint
    assert summary["next_action"] == hint
    # The kernel still names the ready work next to the advice. The earlier approved postpone left none ready.
    assert brief["ready_work"] == ([] if fault == "fresh" else ["Book the room"])
    assert _logical_state(path) == before
    commit_args = {"idempotency_key": "approve-daily-review-plan", "project_id": project.id,
                   "trigger": "owner_request" if str(fault).startswith("owner_bridge") else "scheduled_review", **proposal}
    if fault == "owner_bridge_changed":
        commit_args["summary"] = "The owner selected a different plan."
    elif fault in ("owner_bridge_source_changed", "owner_bridge_no_source"):
        with projects_db.connect_closing(root / "projects.db") as registry:
            if fault == "owner_bridge_source_changed":
                registry.execute("UPDATE project_review_briefs SET source_digest=? WHERE project_id=?",
                                 ("0" * 64, project.id))
            else:
                registry.execute("UPDATE project_review_briefs SET source_task_id=NULL,source_run_id=NULL,source_digest=NULL "
                                 "WHERE project_id=?", (project.id,))
            registry.commit()
    elif fault == "owner_bridge_body_changed":
        with kanban_db.connect_closing(kanban_db.kanban_db_path(board="project-summaries"), board="project-summaries") as board:
            board.execute("UPDATE tasks SET body=body || ? WHERE id=?", ("\nChanged after acceptance.\n", review_task))
            board.commit()
    elif fault == "owner_bridge_disabled":
        (root / "config.yaml").write_text(yaml.safe_dump({"owner_review": {"enabled": False, "allowlist": [project.id]}}))
    elif fault == "owner_bridge_legacy":
        with projects_db.connect_closing(root / "projects.db") as registry:
            for field in ("source_task_id", "source_run_id", "source_digest"):
                registry.execute(f"ALTER TABLE project_review_briefs DROP COLUMN {field}")
            registry.commit()
    with pytest.raises(owner_workspace.OwnerWorkspaceError) as refused:
        owner_workspace.commit_project_plan(ctx, **commit_args)
    assert refused.value.code == "owner_run_authority_required"
    assert _logical_state(path) == before

    approver = _with_approver(ctx.session)
    try:
        committed = _commit_project_plan(ctx, **commit_args)
    finally:
        approver.join()

    assert committed["applied"] is True
    snapshot = owner_workspace.read_project_snapshot(ctx, project.slug, planning_context=True)
    changed = next(item for item in snapshot["planning_context"]["tasks"] if item["id"] == task_id)
    assert changed["status"] == "scheduled"
    assert changed["event_revision"] > target["expected_revision"]
    latest = records.build_review_input(project, events.Cursors(), None)["sources"]["proposals"][0]
    expected_trigger = "owner_request" if str(fault).startswith("owner_bridge_") else "scheduled_review"
    assert latest["trigger"] == expected_trigger
    assert [(item["action"], item["task_id"]) for item in latest["changes"]] == [("postpone", task_id)]
    if str(fault).startswith("owner_bridge"):
        before_replay = _logical_state(path)
        replayed = _commit_project_plan(ctx, **commit_args)
        assert replayed["applied"] is True and _logical_state(path) == before_replay
        assert records.build_review_input(project, events.Cursors(), None)["sources"]["proposals"][0]["trigger"] == expected_trigger
