"""Workspace machine examples (contract map section 5, test 6).

Every example comes from the Kanban plugin's real Workspace machine routes,
read with a locally issued Workspace token over the setup board of
tests/plugins/test_kanban_workspace_machine_read.py. Real kernel calls drive
the board's cards into each review and stopped-work state, and the run
receipts read ended runs written into the board's run table, one per outcome
the receipt reads, each with or without a recorded route and cost. What is
asserted is described in tests/contracts/conftest.py.
"""

from __future__ import annotations

import contextlib
import json
from collections import Counter

import pytest

from hermes_cli import kanban_db as kb, owner_workspace as ow
from plugins.dashboard_auth.raphael_workspace import BOARD
from tests.contracts import conftest as contract
from tests.hermes_cli.test_owner_workspace import _breaker_trips_on, _capability_receipt_metadata
from tests.plugins.test_kanban_workspace_machine_read import (  # noqa: F401  (workspace_surface is a fixture)
    _break_board_open,
    _get,
    workspace_surface,
)

FAMILY = "workspace_machine"
QUERY = f"?board={BOARD}"
RESET_AT = 1_790_179_200  # two hours after the runs below end


def _capability(source: str) -> dict:
    return {
        "schema_version": 1, "skills": [], "skills_truncated": False, "tools": ["read_file"],
        "tools_truncated": False, "connections": [], "connections_truncated": False,
        "source": source, "truncated": False,
    }


def _costed(state: str) -> dict:
    metadata = _capability_receipt_metadata(2)
    metadata["runtime_receipt"]["cost"]["state"] = state
    return metadata


ENDED_RUNS = (
    ("completed", _costed("estimated")),
    ("review_requested", _costed("exact")),
    ("scheduled", _costed("reported")),
    ("crashed", _costed("included")),
    ("completed_unreported", _capability_receipt_metadata(3, _capability("session-tool-calls"))),
    ("rate_limited", _capability_receipt_metadata(3, _capability("unavailable"))),
    ("rate_limited", {"rate_limit_reset_at": RESET_AT}),
)


def _card(project_id: str, title: str, **fields) -> str:
    with contextlib.closing(kb.connect(board=BOARD)) as conn:
        return kb.create_task(
            conn, title=title, assignee="default", project_id=project_id, **fields,
        )


def _claimable(project_id: str, title: str) -> str:
    """A read-only card. The setup board's running card holds the whole
    repository, which defers the claim of any card that could write to it."""
    return _card(project_id, title, owned_paths=[])


def _record_run(conn, task_id: str, outcome: str, metadata: dict) -> int:
    """Write one ended run as the kernel's run table holds it."""
    return conn.execute(
        "INSERT INTO task_runs (task_id, profile, status, started_at, ended_at, outcome,"
        " metadata) VALUES (?, 'raphael-verifier', ?, 1790171000, 1790172000, ?, ?)",
        (task_id, outcome, outcome, json.dumps(metadata)),
    ).lastrowid


def _read(surface: dict, path: str, query: str = QUERY) -> dict:
    response = _get(surface, f"/api/plugins/kanban{path}", query=query)
    return {"status": response.status_code, "body": response.json()}


def test_the_workspace_machine_examples_carry_every_route_state_and_error(
    workspace_surface, owner_payload_example,
):
    s = workspace_surface
    approved = _card(s["project_id"], "Review the approved outline")
    with contextlib.closing(kb.connect(board=BOARD)) as conn:
        assert kb.request_review(conn, approved, summary="Ready.")
        assert kb.complete_task(conn, approved, result="Approved.")
    # The breaker trips on every ready or review card, so this comes next.
    _breaker_trips_on(BOARD, _claimable(s["project_id"], "Print the workshop handouts"))
    capability = _claimable(s["project_id"], "Connect the payment provider")
    with contextlib.closing(kb.connect(board=BOARD)) as conn:
        assert kb.claim_task(conn, capability) is not None
        assert kb.block_task(
            conn, capability, reason="the provider account has no API credentials",
            kind="capability", expected_run_id=kb.get_task(conn, capability).current_run_id,
        )
    reviews = {
        state: _card(s["project_id"], f"Review the {state} outline")
        for state in ("awaiting", "changes")
    }
    recorded = _card(s["project_id"], "Record the workshop runs")
    tiered = {
        tier: _card(s["project_id"], f"Tier {tier} work", owned_paths=[], risk_tier=tier)
        for tier in (0, 1, 2)
    }
    with contextlib.closing(kb.connect(board=BOARD)) as conn:
        kb._append_event(
            conn, tiered[1], "risk_tier_raised",
            {"from": 0, "to": 1, "reviewer": "default", "run_id": 1},
        )
        for task_id in reviews.values():
            assert kb.request_review(conn, task_id, summary="Ready.")
        assert kb.claim_review_task(conn, reviews["changes"]) is not None
        assert kb.request_changes(conn, reviews["changes"], reason="Name the date first.")[0]
        run_ids = [_record_run(conn, recorded, *run) for run in ENDED_RUNS]
        conn.commit()

    live = {
        "projects": _read(s, "/projects", ""),
        "projects_with_removal_state": _read(
            s, "/projects", f"?capabilities={ow.OWNER_PROJECT_REMOVAL_STATE_CAPABILITY}"),
        "profiles": _read(s, "/profiles", ""),
        "boards": _read(s, "/boards", ""),
        "board": _read(s, "/board"),
        "workers": _read(s, "/workers/active"),
        "assignees": _read(s, "/assignees"),
        "task_runs": _read(s, f"/tasks/{recorded}"),
        "task_attachments": _read(s, f"/tasks/{s['ready_id']}/attachments"),
        "run_receipts": [_read(s, f"/runs/{run_id}") for run_id in (s["run_id"], *run_ids)],
        "board_bad_request": _read(s, "/board", f"{QUERY}&capabilities=run_task_context"),
        "run_not_found": _read(s, "/runs/9999"),
    }
    with pytest.MonkeyPatch.context() as broken:
        _break_board_open(broken)
        live["board_unavailable"] = _read(s, "/board")
    # An absent board database is a board with no work, not an outage.
    kb.kanban_db_path(BOARD).unlink()
    live["board_empty"] = _read(s, "/board")

    saved = {kind: owner_payload_example(FAMILY, kind, answer) for kind, answer in live.items()}

    assert live["task_runs"]["body"]["runs"] == run_ids
    for answers in (live, saved):
        columns = answers["board"]["body"]["columns"]
        names = [column["name"] for column in columns]
        empty = answers["board_empty"]["body"]["columns"]
        assert [column["name"] for column in empty] == names
        assert not any(column["tasks"] for column in empty)
        [board] = answers["boards"]["body"]["boards"]
        assert set(board["counts"]) == {*names, "archived"}
        assert [board["counts"][name] for name in names] == [len(c["tasks"]) for c in columns]
        assert board["total"] == sum(len(column["tasks"]) for column in columns)
        [project] = answers["projects"]["body"]["projects"]
        [with_removal] = answers["projects_with_removal_state"]["body"]["projects"]
        assert board["project_id"] == project["id"] == with_removal["id"]
        assert set(with_removal) - set(project) == {"removal_state"}
        tasks = [(column["name"], task) for column in columns for task in column["tasks"]]
        assert Counter((task["assignee_name"], name) for name, task in tasks) == {
            (assignee["name"], status): count
            for assignee in answers["assignees"]["body"]["assignees"]
            for status, count in assignee["counts"].items()
        }
        assert all(type(task["risk_tier_raised"]) is bool for _name, task in tasks)
        assert contract.closed_values(answers["board"]["body"], ("risk_tier",)) == {0, 1, 2, None}
        assert {task["risk_tier"] for _name, task in tasks if task["risk_tier_raised"]} == {1}
        titles = {worker["task_title"] for worker in answers["workers"]["body"]["workers"]}
        assert titles and titles <= {task["title"] for name, task in tasks if name == "running"}
        [attachment] = answers["task_attachments"]["body"]["attachments"]
        assert attachment["size"] == len(s["attachment_body"])
        runs = [answer["body"]["run"] for answer in answers["run_receipts"]]
        assert len(runs) == len(answers["task_runs"]["body"]["runs"]) + 1
        assert [run["finished_at"] is None for run in runs] == [
            run["receipt"]["outcome"] == "running" for run in runs]
        # Only a version 3 route receipt carries the capability object.
        assert ["capability" in run["receipt"]["runtime"] for run in runs[1:]] == [
            metadata.get("runtime_receipt", {}).get("schema_version") == 3
            for _outcome, metadata in ENDED_RUNS
        ]
        single = [answer for kind, answer in answers.items() if kind != "run_receipts"]
        for answer in single + answers["run_receipts"]:
            assert (answer["status"] >= 400) == ("detail" in answer["body"])


def test_the_workspace_machine_examples_keep_to_every_closed_vocabulary():
    """The saved Workspace machine examples, read together, take every value of
    each row of the table; the fixture holds each one to its row."""
    contract.assert_closed_vocabularies_covered(FAMILY)
