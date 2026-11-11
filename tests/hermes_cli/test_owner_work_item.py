"""Build P2 of the plan of 31 October: the Project snapshot carries each card's and each run's work item.

Only a reader that names ``work_item_v1`` gets ``work_item`` on every card and every run: the key of the work the
card or run is an attempt at, and whether it is the item's current attempt. A card is current as
hermes_cli/kanban_work_items.py decides; a run is current when its card is and no newer run of that card is listed.
The platform adds no grouping object and no text: an item's earlier attempts are the carried cards and runs that
share its key with ``current`` false, and its current attempt is the one row with ``current`` true. Every other
reader gets today's snapshot exactly.

Real boards of a committed owner Project, read through read_project_snapshot, and the real snapshot route of the
api_server. Retries are the kernel's own claims and reclaims, and the owner's replace, split and merge go through
apply_owner_project_plan. The records written here by hand are the ones P1's tests write: the
review_followup_recorded event _continuation records on a reworked card, since a real review return needs the whole
delivery step, and the delivery_updated event naming an update card on main, which needs a delivery and a moved
main; and, for the 200-card window, the statuses and the created and event times of the finished cards that fill
it, as the window tests set them, so the predecessor is the oldest finished card.
"""

from __future__ import annotations

import json
import time
from types import SimpleNamespace
from unittest.mock import ANY, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from hermes_cli import kanban_db as kb, owner_workspace as ow
from tests.hermes_cli.test_kanban_work_items import KEY, _continued_by
from tests.hermes_cli.test_owner_delivery_state import (  # noqa: F401  (owner is a fixture)
    _TODAYS_CARD_KEYS, _board, _cards, _project, owner,
)
from tests.hermes_cli.test_owner_plan_add_after_replace import _apply, _replace, _spec
from tests.hermes_cli.test_owner_snapshot_task_window import _add_tasks

_TODAYS_RUN_KEYS = {"started_at", "finished_at", "receipt"}
_RUN_CONTEXT_KEYS = {"task_title", "has_newer_run", "retry_origin"}
_SNAPSHOT = "/v1/owner-workspace/projects/{project_slug}/snapshot"


def _card(conn, project: dict, title: str, **fields) -> str:
    return kb.create_task(conn, title=title, assignee="raphael-worker", project_id=project["project_id"], **fields)


def _retried(conn, card: str, times: int) -> None:
    """``card`` reclaimed ``times`` times and claimed again: one run per claim, the newest still running."""
    for _ in range(times):
        assert kb.claim_task(conn, card) is not None
        assert kb.reclaim_task(conn, card, reason="try again") is True
    assert kb.claim_task(conn, card) is not None


def _updated(conn, source: str, update: str) -> None:
    """``source``'s delivery_updated event naming ``update``, its update card on main."""
    with kb.write_txn(conn):
        kb._append_event(conn, source, "delivery_updated", {
            "delivery_id": 1, "head": "a" * 40, "observed_main": "b" * 40, "update_task_id": update,
            "update_head": "c" * 40, "update_branch": f"delivery-update/{source}"})


def _plan(conn, project: dict) -> SimpleNamespace:
    """The Project as apply_owner_project_plan names it: its board, its id and the anchor its bootstrap made."""
    (anchor,) = (row[0] for row in conn.execute(
        "SELECT id FROM tasks WHERE project_id = ? AND task_kind = 'control'", (project["project_id"],)))
    return SimpleNamespace(conn=conn, id=project["project_id"], anchor_id=anchor)


def _target(conn, card: str) -> dict:
    """The snapshot an owner plan names ``card`` by."""
    return {"task_id": card, "expected_status": kb.get_task(conn, card).status,
            "expected_revision": kb.owner_plan_task_revision(conn, card)}


def _snapshot(owner: ow.OwnerContext, project: dict, **capabilities) -> dict:
    return ow.read_project_snapshot(owner, project["slug"], run_context=True, **capabilities)


def _rows(snapshot: dict) -> list[dict]:
    """Every card, then every run, of ``snapshot``."""
    return [*(task for column in snapshot["columns"] for task in column["tasks"]), *snapshot["runs"]]


def _runs(snapshot: dict, title: str) -> list[tuple]:
    """The work item of each run of the card titled ``title``, newest first, as (key, current)."""
    return [(run["work_item"]["key"], run["work_item"]["current"])
            for run in snapshot["runs"] if run["task_title"] == title]


def _grouped(rows: list[dict]) -> dict:
    """``rows`` grouped as a reader groups them: each key to the titles of its current rows and of its earlier
    ones, in payload order."""
    items: dict = {}
    for row in rows:
        current, earlier = items.setdefault(row["work_item"]["key"], ([], []))
        (current if row["work_item"]["current"] else earlier).append(row.get("title", row.get("task_title")))
    return items


def _attempts(owner: ow.OwnerContext) -> tuple[dict, dict]:
    """A Project holding each kind of attempt the snapshot carries, and its cards by name: a card retried twice, a
    reworked card and its continuation, an update card on main, a card the owner replaced after one run, and a card
    of its own."""
    project = _project(owner, "Attempts Pilot")
    with _board(project) as conn:
        cards = {"retried": _card(conn, project, "Write the launch copy")}
        _retried(conn, cards["retried"], 2)
        cards["source"] = _card(conn, project, "Build the signup form")
        _retried(conn, cards["source"], 0)
        assert kb.complete_task(conn, cards["source"], result="Returned for rework.") is True
        cards["rework"] = _card(conn, project, "Rework the signup form")
        _continued_by(conn, cards["source"], cards["rework"])
        _retried(conn, cards["rework"], 1)
        cards["shipped"] = _card(conn, project, "Publish the pricing page")
        assert kb.complete_task(conn, cards["shipped"], result="Approved.") is True
        cards["update"] = _card(conn, project, "Bring the pricing page up to date", parents=[cards["shipped"]])
        _updated(conn, cards["shipped"], cards["update"])
        cards["replaced"] = _card(conn, project, "Write the content")
        assert kb.claim_task(conn, cards["replaced"]) is not None
        assert kb.reclaim_task(conn, cards["replaced"], reason="try again") is True
        (cards["replacement"],) = _apply(_plan(conn, project), [_replace(_target(conn, cards["replaced"]))])
        cards["alone"] = _card(conn, project, "Book the photographer")
        assert kb.get_task(conn, cards["replaced"]).status == "archived"
    return project, cards


def test_a_work_item_reader_gets_a_work_item_on_every_card_and_every_run(owner):
    assert ow.WORK_ITEM_CAPABILITY == "work_item_v1"
    project, _ = _attempts(owner)

    snapshot = _snapshot(owner, project, work_item=True)

    rows = _rows(snapshot)
    assert len(rows) == 7 + 8  # the carried cards and every run, the replaced card's own included
    for row in rows:
        assert set(row["work_item"]) == {"key", "current"}
        assert KEY.fullmatch(row["work_item"]["key"])
        assert type(row["work_item"]["current"]) is bool


@pytest.mark.parametrize("run_context", [False, True])
def test_a_reader_without_the_capability_gets_todays_snapshot_exactly(owner, monkeypatch, run_context):
    """The same board read with and without the capability, at one frozen moment: today's card and run keys, no
    work_item anywhere, and byte for byte the capability snapshot without its work items."""
    project, _ = _attempts(owner)
    now = time.time()
    monkeypatch.setattr(time, "time", lambda: now)

    today = ow.read_project_snapshot(owner, project["slug"], run_context=run_context)
    with_items = ow.read_project_snapshot(owner, project["slug"], run_context=run_context, work_item=True)

    tasks = [task for column in today["columns"] for task in column["tasks"]]
    assert tasks and all(set(task) == _TODAYS_CARD_KEYS for task in tasks)
    assert today["runs"] and all(
        set(run) == _TODAYS_RUN_KEYS | (_RUN_CONTEXT_KEYS if run_context else set()) for run in today["runs"])
    assert '"work_item"' not in json.dumps(today)
    for row in _rows(with_items):
        del row["work_item"]
    assert json.dumps(today, sort_keys=True) == json.dumps(with_items, sort_keys=True)


def test_a_card_retried_twice_has_three_runs_of_its_key_and_only_the_newest_is_current(owner):
    project = _project(owner, "Retry Pilot")
    with _board(project) as conn:
        _retried(conn, _card(conn, project, "Write the launch copy"), 2)

    snapshot = _snapshot(owner, project, work_item=True)

    card = _cards(snapshot)["Write the launch copy"]
    key = card["work_item"]["key"]
    assert card["work_item"] == {"key": key, "current": True}
    assert _runs(snapshot, "Write the launch copy") == [(key, True), (key, False), (key, False)]
    assert [run["has_newer_run"] for run in snapshot["runs"]] == [False, True, True]
    assert _grouped(snapshot["runs"]) == {key: (["Write the launch copy"], ["Write the launch copy"] * 2)}


def test_a_rework_after_review_is_the_current_attempt_and_its_source_an_earlier_one(owner):
    """The source's runs and the continuation's share the source's key; only the continuation's newest run is
    current, though the source's newest run has no newer run of its own card."""
    project = _project(owner, "Rework Pilot")
    with _board(project) as conn:
        source = _card(conn, project, "Build the signup form")
        _retried(conn, source, 1)
        assert kb.complete_task(conn, source, result="Returned for rework.") is True
        rework = _card(conn, project, "Rework the signup form")
        _continued_by(conn, source, rework)
        _retried(conn, rework, 1)

    snapshot = _snapshot(owner, project, work_item=True)

    cards = _cards(snapshot)
    key = cards["Build the signup form"]["work_item"]["key"]
    assert cards["Build the signup form"]["work_item"] == {"key": key, "current": False}
    assert cards["Rework the signup form"]["work_item"] == {"key": key, "current": True}
    assert _runs(snapshot, "Rework the signup form") == [(key, True), (key, False)]
    assert _runs(snapshot, "Build the signup form") == [(key, False), (key, False)]
    assert [run["has_newer_run"] for run in snapshot["runs"]] == [False, True, False, True]
    assert _grouped([task for column in snapshot["columns"] for task in column["tasks"]]) == {
        key: (["Rework the signup form"], ["Build the signup form"])}
    assert _grouped(snapshot["runs"]) == {key: (
        ["Rework the signup form"], ["Rework the signup form", "Build the signup form", "Build the signup form"])}


def test_an_owner_replacement_carries_the_replaced_cards_runs_as_earlier_attempts(owner):
    """The archived card is not in the columns, but its run is listed with the replacement's key."""
    project = _project(owner, "Replace Pilot")
    with _board(project) as conn:
        replaced = _card(conn, project, "Write the content")
        assert kb.claim_task(conn, replaced) is not None
        assert kb.reclaim_task(conn, replaced, reason="try again") is True
        (replacement,) = _apply(_plan(conn, project), [_replace(_target(conn, replaced))])
        assert kb.get_task(conn, replaced).status == "archived"

    snapshot = _snapshot(owner, project, work_item=True)

    cards = _cards(snapshot)
    assert "Write the content" not in cards
    assert cards["Rewrite the content"]["id"] == replacement
    key = cards["Rewrite the content"]["work_item"]["key"]
    assert cards["Rewrite the content"]["work_item"] == {"key": key, "current": True}
    assert _runs(snapshot, "Write the content") == [(key, False)]
    assert snapshot["runs"][0]["has_newer_run"] is False


def test_an_update_card_on_main_has_its_sources_key_and_is_current(owner):
    project = _project(owner, "Update Pilot")
    with _board(project) as conn:
        source = _card(conn, project, "Publish the pricing page")
        assert kb.complete_task(conn, source, result="Approved.") is True
        _updated(conn, source, _card(conn, project, "Bring the pricing page up to date", parents=[source]))

    snapshot = _snapshot(owner, project, work_item=True)

    cards = _cards(snapshot)
    key = cards["Publish the pricing page"]["work_item"]["key"]
    assert cards["Publish the pricing page"]["work_item"] == {"key": key, "current": False}
    assert cards["Bring the pricing page up to date"]["work_item"] == {"key": key, "current": True}
    assert _runs(snapshot, "Publish the pricing page") == [(key, False)]


def test_a_split_and_a_merge_by_the_owner_each_start_a_new_current_item(owner):
    originals = ("Write the content", "Write the home page", "Write the about page")
    project = _project(owner, "Plan Pilot")
    with _board(project) as conn:
        content, home, about = (_card(conn, project, title) for title in originals)
    before = _cards(_snapshot(owner, project, work_item=True))
    with _board(project) as conn:
        _apply(_plan(conn, project), [{
            "action": "split", "reason": "The content is too broad to verify safely.",
            "target": _target(conn, content), "replacements": [
                {**_spec("Draft the documents"), "parents": []}, {**_spec("Translate the documents"), "parents": []}],
        }, {
            "action": "merge", "reason": "One coherent set of pages is easier to verify.",
            "targets": [_target(conn, home), _target(conn, about)], "replacement": _spec("Write the pages"),
        }])

    cards = _cards(_snapshot(owner, project, work_item=True))

    assert not cards.keys() & set(originals)
    made = [cards[title]["work_item"] for title in ("Draft the documents", "Translate the documents", "Write the pages")]
    assert all(item["current"] is True for item in made)
    assert len({item["key"] for item in made} | {before[title]["work_item"]["key"] for title in originals}) == 6


def test_a_continuation_whose_predecessor_is_outside_the_window_has_its_key_and_is_current(owner):
    project = _project(owner, "Window Pilot")
    with _board(project) as conn:
        predecessor = _card(conn, project, "Build the signup form")
        assert kb.complete_task(conn, predecessor, result="Returned for rework.") is True
        _continued_by(conn, predecessor, _card(conn, project, "Rework the signup form"))
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET created_at = 500 WHERE id = ?", (predecessor,))
            conn.execute("UPDATE task_events SET created_at = 500 WHERE task_id = ?", (predecessor,))
    key = _cards(_snapshot(owner, project, work_item=True))["Build the signup form"]["work_item"]["key"]
    _add_tasks(project, 205, status="done", created_start=1_000)

    snapshot = _snapshot(owner, project, work_item=True)

    cards = _cards(snapshot)
    assert snapshot["truncated"]["tasks"] is True
    assert "Build the signup form" not in cards
    assert cards["Rework the signup form"]["work_item"] == {"key": key, "current": True}
    assert _runs(snapshot, "Build the signup form") == [(key, False)]


def test_no_card_id_appears_in_the_runs_of_a_work_item_reader(owner):
    project, cards = _attempts(owner)
    with _board(project) as conn:
        ids = [row[0] for row in conn.execute("SELECT id FROM tasks WHERE project_id = ?", (project["project_id"],))]
    assert set(cards.values()) < set(ids)

    runs = _snapshot(owner, project, work_item=True)["runs"]

    assert "Write the content" in {run["task_title"] for run in runs}  # the archived card's run is listed
    assert not any(task_id in json.dumps(runs) for task_id in ids)


def _app() -> web.Application:
    """The adapter's own snapshot route, as ``connect()`` adds it."""
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={}))
    app = web.Application()
    for method, path, handler in adapter._http_route_table():
        if method == "GET" and path == _SNAPSHOT:
            app.router.add_route(method, path, handler)
    return app


@pytest.mark.asyncio
@pytest.mark.parametrize("query, run_context, forwarded", [
    ("", False, {}),
    ("?capabilities=work_item_v1", False, {"work_item": True}),
    ("?capabilities=run_task_context,work_item_v1", True, {"work_item": True}),
    ("?capabilities=delivery_outcome_v1,work_item_v1", False, {"delivery_outcome": True, "work_item": True}),
    ("?capabilities=delivery_outcome_v1", False, {"delivery_outcome": True}),
    ("?capabilities=work_item_v2", False, {}),
    ("?capabilities=work_item", False, {}),
    ("?capabilities=work_item_v1&capabilities=work_item_v1", False, {}),
    ("?capabilities=delivery_outcome_v1&capabilities=work_item_v1", False, {}),
])
async def test_the_snapshot_forwards_work_item_only_when_asked(query, run_context, forwarded):
    """Only a reader that names ``work_item_v1`` in ONE ``capabilities`` parameter makes the kernel call with
    ``work_item=True``; every other request makes exactly today's call."""
    config = {"gateway": {"api_server": {"owner_workspace": {"enabled": True}}}}
    with (
        patch("gateway.run._load_gateway_config", return_value=config),
        patch("hermes_cli.owner_workspace.resolve_owner_context", return_value=object()),
        patch("hermes_cli.owner_workspace.read_project_snapshot", return_value={}) as read,
    ):
        async with TestClient(TestServer(_app())) as client:
            resp = await client.get(f"/v1/owner-workspace/projects/shoe-shop/snapshot{query}")

    assert resp.status == 200
    read.assert_called_once_with(ANY, "shoe-shop", run_context=run_context, planning_context=False, **forwarded)
