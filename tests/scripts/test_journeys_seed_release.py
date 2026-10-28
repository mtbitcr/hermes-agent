"""Tests for the journeys seed, card 3: journey 2's review cards, the approved card shown as being delivered, and its one
waiting release batch.

seed_states.py runs for real against a temporary platform home, as in test_journeys_seed.py. The reads are the
real ones: the owner-workspace board read, and the release decision route of the API server.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import hashlib
import importlib.util
import json
import os
import re
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path

import pytest
import yaml
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from hermes_cli import kanban_db, owner_workspace as ow, release_ledger, release_unit

pytestmark = pytest.mark.linux_only

REPO = Path(__file__).resolve().parents[2]
SEED_STATES = REPO / "scripts" / "journeys" / "seed_states.py"
STACK_HOME = REPO / "scripts" / "journeys" / "stack-home"

BOARD = "journey-test"
PROJECT_SLUG = "journey-test-project"
PREFIX = "Journey test"
PROJECT_NAME = "Journey test project"
REVIEW = "Journey test card waiting for review"
APPROVED = "Journey test card approved"
MERGED = "Journey test card merged"
LEDGER = "kanban/release_ledger.db"
VIEW = "/v1/owner-workspace/release"
# The columns of kanban_deliveries that record_approved_delivery leaves empty: the row has no pull request and no merge.
UNSET_DELIVERY_COLUMNS = (
    "integration_task_id",
    "pull_request_number",
    "pull_request_head",
    "pull_request_state",
    "pull_request_branch",
    "publish_lease",
    "publish_lease_until",
    "publish_refusal",
)

SEED = {
    "prefix": PREFIX,
    "journeys": {
        "2": {
            "projects": [
                {
                    "name": PROJECT_NAME,
                    "questions": [],
                    "suggestions": [],
                    "cards": [{"name": REVIEW}, {"name": APPROVED}, {"name": MERGED}],
                    "release_batch": {"name": "Journey test release batch", "cards": [MERGED]},
                }
            ]
        }
    },
}
SEED_TEXT = json.dumps(SEED, indent=2) + "\n"

# Bytecode of the platform modules that the seed's commands import is cached here, never in the tree.
REAL_PYC = Path(tempfile.gettempdir()) / f"journeys-seed-pyc-{os.getuid()}"  # windows-footgun: ok - the module is linux_only


def tree(root: Path) -> dict[str, str]:
    """Relative path -> content hash. SQLite's own -wal and -shm files are left out (owner rule 1)."""
    seen: dict[str, str] = {}
    for base, dirs, files in os.walk(root):
        for name in sorted(dirs) + sorted(f for f in files if not f.endswith(("-wal", "-shm"))):
            full = Path(base) / name
            seen[str(full.relative_to(root))] = "dir" if full.is_dir() else hashlib.sha256(full.read_bytes()).hexdigest()
    return seen


def rows_of(root: Path) -> dict[str, dict[str, list[tuple]]]:
    """File -> table -> every row, for each SQLite file under root, read through a read-only connection."""
    found: dict[str, dict[str, list[tuple]]] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        with path.open("rb") as handle:
            if handle.read(16) != b"SQLite format 3\x00":
                continue
        with contextlib.closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as conn:
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND sql NOT LIKE 'CREATE VIRTUAL TABLE%' ORDER BY name"
            ).fetchall()
            found[str(path.relative_to(root))] = {
                name: sorted(conn.execute('SELECT * FROM "%s"' % name.replace('"', '""')).fetchall(), key=repr)
                for (name,) in tables
            }
    return found


def snapshot(root: Path) -> dict[str, dict]:
    """What owner rule 1 means by writing nothing: the store files' content hashes and every row of every SQLite store."""
    return {"files": tree(root), "rows": rows_of(root)}


def ledger_rows(home: Path, sql: str) -> list[tuple]:
    with contextlib.closing(sqlite3.connect((home / LEDGER).resolve().as_uri() + "?mode=ro", uri=True)) as conn:
        return conn.execute(sql).fetchall()


def board_rows(sql: str, *args) -> list[dict]:
    """Rows of the journey board file, read through a read-only connection."""
    path = kanban_db.kanban_db_path(BOARD)
    with contextlib.closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(row) for row in conn.execute(sql, args)]


def edited(change) -> str:
    data = copy.deepcopy(SEED)
    change(data)
    return json.dumps(data)


def cards_of(data: dict) -> list:
    return data["journeys"]["2"]["projects"][0]["cards"]


@pytest.fixture
def seed(monkeypatch):
    """seed_states.py imported for real; the commands it runs find the platform and keep bytecode in /tmp."""
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    monkeypatch.setenv("PYTHONPATH", str(REPO))
    monkeypatch.setenv("PYTHONPYCACHEPREFIX", str(REAL_PYC))
    spec = importlib.util.spec_from_file_location("journeys_seed_states", SEED_STATES)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def reset_home() -> Path:
    """A freshly reset stack home: what reset_stack.sh writes from stack-home, and nothing else."""
    home = Path(os.environ["HERMES_HOME"])
    shutil.copytree(STACK_HOME, home, dirs_exist_ok=True)
    return home


@pytest.fixture
def no_units(monkeypatch):
    """No release unit is listed or started: the page reads and the put off must not start anything."""

    def started(*argv, **options):
        raise AssertionError("a process was started")

    monkeypatch.setattr(release_unit, "busy_units", lambda: [])
    monkeypatch.setattr(asyncio, "create_subprocess_exec", started)


def run_seed(seed, capsys, journey: str, text: str = SEED_TEXT) -> tuple[int, str, str]:
    code = seed.main([journey], text)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def seed_journey_2(seed, capsys) -> str:
    code, out, err = run_seed(seed, capsys, "2")
    assert code == 0, err
    return out


def board_cards() -> dict[str, tuple[str, dict]]:
    """Title -> (column, card), from the owner-workspace board read of the journey Project.

    The read is made as a platform process that opens the journey board for the first time after the seed: that first
    open binds the cards to the Project and parks every card in review that has no approved model route.
    """
    kanban_db._INITIALIZED_PATHS.clear()
    with contextlib.closing(kanban_db.connect(board=BOARD)):
        pass
    read = ow.read_project_snapshot(ow.resolve_owner_context(), PROJECT_SLUG, delivery_state=True)
    return {task["title"]: (column["name"], task) for column in read["columns"] for task in column["tasks"]}


def enable_owner_workspace(home: Path, monkeypatch) -> None:
    """The owner workspace routes are served only when the home's config enables them."""
    monkeypatch.setattr("gateway.run._hermes_home", home)
    config = home / "config.yaml"
    data = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
    data.setdefault("gateway", {}).setdefault("api_server", {})["owner_workspace"] = {"enabled": True}
    config.write_text(yaml.safe_dump(data), encoding="utf-8")


def client() -> TestClient:
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={}))
    app = web.Application()
    for method, path, handler in adapter._http_route_table():
        if path.startswith(VIEW):
            app.router.add_route(method, path, handler)
    return TestClient(TestServer(app))


# --- 1. The board read: the review card and the approved card.


def test_the_board_read_gives_the_review_card_as_awaiting_review_and_the_approved_card_as_approved(
    seed, reset_home, capsys,
):
    seed_journey_2(seed, capsys)

    cards = board_cards()
    assert set(cards) == {REVIEW, APPROVED, MERGED}
    assert cards[REVIEW][1]["review_state"] == "awaiting_review"
    assert cards[APPROVED][1]["review_state"] == "approved"


def test_the_approved_card_is_in_a_group_the_plan_page_shows_open(seed, reset_home, capsys):
    seed_journey_2(seed, capsys)

    column, approved = board_cards()[APPROVED]
    # The Plan page shows a done card under the closed Completed section unless its delivery is unfinished; then it shows
    # the card as being delivered, under Now.
    assert (column, approved["review_state"], approved["delivery"]) == ("done", "approved", "delivering")

    # The delivery is the one row record_approved_delivery would insert: no pull request and no merge, and none for the
    # other two cards.
    rows = board_rows("SELECT * FROM kanban_deliveries")
    assert [row["source_task_id"] for row in rows] == [approved["id"]]
    [row] = rows
    [completed] = board_rows(
        "SELECT run_id FROM task_events WHERE task_id = ? AND kind = 'completed' ORDER BY id DESC LIMIT 1", approved["id"]
    )
    assert re.fullmatch(r"[0-9a-f]{40}", row["source_head"]), row["source_head"]
    assert completed["run_id"] is not None
    assert row["approval_run_id"] == completed["run_id"]
    assert isinstance(row["created_at"], int)
    assert {name: row[name] for name in UNSET_DELIVERY_COLUMNS} == dict.fromkeys(UNSET_DELIVERY_COLUMNS)


# --- 2. The release decision read, and the Put off.


@pytest.mark.asyncio
async def test_the_release_decision_read_gives_one_waiting_batch_holding_the_merged_card(
    seed, reset_home, capsys, monkeypatch, no_units,
):
    seed_journey_2(seed, capsys)
    enable_owner_workspace(reset_home, monkeypatch)

    async with client() as http:
        response = await http.get(VIEW)
        body = await response.json()

    assert response.status == 200
    waiting = body["waiting"]
    assert (waiting["count"], waiting["titles"], waiting["actions"]) == (1, [MERGED], ["accept", "defer"])
    assert body["release"] is None
    batches = ledger_rows(reset_home, "SELECT COUNT(*) FROM release_batches WHERE state IN ('open', 'deferred')")
    assert batches == [(1,)]
    merged_id = board_cards()[MERGED][1]["id"]
    assert ledger_rows(reset_home, "SELECT card_id, title, board FROM release_members") == [(merged_id, MERGED, BOARD)]


@pytest.mark.asyncio
async def test_a_put_off_on_the_batch_succeeds_once_and_changes_nothing_else(
    seed, reset_home, capsys, monkeypatch, no_units,
):
    seed_journey_2(seed, capsys)
    enable_owner_workspace(reset_home, monkeypatch)

    async with client() as http:
        shown = (await (await http.get(VIEW)).json())["waiting"]
        before = snapshot(reset_home)
        assert ledger_rows(reset_home, "SELECT state, version FROM release_batches") == [("open", shown["version"])]
        assert ledger_rows(reset_home, "SELECT COUNT(*) FROM release_events") == [(0,)]

        defer = f"{VIEW}/{shown['batch_id']}/defer"
        response = await http.post(defer, json={"version": shown["version"], "digest": shown["digest"]})
        body = await response.json()
        after = snapshot(reset_home)
        repeated = await http.post(defer, json={"version": shown["version"], "digest": shown["digest"]})

    assert response.status == 200
    assert body["answer"] == "put_off"
    # Only the release record changed: the batch is put off, and one answer is recorded.
    assert ledger_rows(reset_home, "SELECT state, version FROM release_batches") == [("deferred", shown["version"] + 1)]
    assert ledger_rows(reset_home, "SELECT kind FROM release_events") == [("release_decided",)]
    assert after["files"].keys() == before["files"].keys()
    assert {name for name in after["files"] if after["files"][name] != before["files"][name]} == {LEDGER}
    for name, tables in before["rows"].items():
        for table, rows in tables.items():
            if (name, table) not in {(LEDGER, "release_batches"), (LEDGER, "release_events"), (LEDGER, "sqlite_sequence")}:
                assert after["rows"][name][table] == rows, (name, table)
    # The second put off is refused and records no second answer.
    assert repeated.status == 409
    assert snapshot(reset_home) == after


# --- 3. A release record that already exists is refused.


def _an_empty_release_record() -> None:
    release_ledger.connect().close()


def _a_release_record_with_a_merge() -> None:
    conn = release_ledger.connect()
    try:
        release_ledger.record_merge(
            conn,
            merge_commit="1" * 40,
            pr_url="https://github.com/mtbitcr/hermes-agent/pull/1",
            reviewed_base="2" * 40,
            reviewed_head="3" * 40,
            reviewed_tree="4" * 40,
            tier=1,
            card_id="t_00000001",
            title="A change that was already there",
            on_main=True,
        )
    finally:
        conn.close()


@pytest.mark.parametrize("make_record", [_an_empty_release_record, _a_release_record_with_a_merge])
def test_the_seed_refuses_when_a_release_record_already_exists_and_writes_nothing(
    seed, reset_home, capsys, make_record,
):
    make_record()
    before = snapshot(reset_home)

    code, out, err = run_seed(seed, capsys, "2")

    assert (code, out) == (1, "")
    assert err == "seed_states: a release record already exists; nothing was written\n"
    assert snapshot(reset_home) == before
    assert not (reset_home / "projects.db").exists()
    assert not (reset_home / "kanban" / "boards" / BOARD).exists()


# --- 4. The made-up merge passes the release record's own checks and is not shown.


@pytest.mark.asyncio
async def test_the_made_up_commit_ids_and_the_link_pass_the_record_checks_and_appear_nowhere_on_the_page(
    seed, reset_home, capsys, monkeypatch, no_units, tmp_path,
):
    out = seed_journey_2(seed, capsys)
    enable_owner_workspace(reset_home, monkeypatch)
    [(merge, link, base, head, tree_id, tier, card_id, title, board)] = ledger_rows(
        reset_home,
        "SELECT merge_commit, pr_url, reviewed_base, reviewed_head, reviewed_tree, tier, card_id, title, board"
        " FROM release_members",
    )
    [delivery] = board_rows("SELECT source_head FROM kanban_deliveries")
    source_head = delivery["source_head"]
    assert len({merge, base, head, tree_id}) == 4
    assert source_head not in {merge, base, head, tree_id}

    async with client() as http:
        page = await (await http.get(VIEW)).text()
    everything = "\n".join(
        [
            page,
            out,
            json.dumps(board_cards()),
            json.dumps(ow.list_owner_decisions(ow.resolve_owner_context())),
        ]
    )
    commits = (merge, base, head, tree_id, source_head)
    shown = [*commits, link, "github.com", "pull/"] + [commit[:7] for commit in commits]
    assert [item for item in shown if item in everything] == []

    # The same values are taken by record_merge on a record of their own, so its own checks accept them.
    elsewhere = tmp_path / "another-home"
    elsewhere.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(elsewhere))
    conn = release_ledger.connect()
    try:
        recorded = release_ledger.record_merge(
            conn,
            merge_commit=merge,
            pr_url=link,
            reviewed_base=base,
            reviewed_head=head,
            reviewed_tree=tree_id,
            tier=tier,
            card_id=card_id,
            title=title,
            board=board,
            on_main=True,
        )
    finally:
        conn.close()
    assert recorded["recorded"] is True
    assert recorded["member"]["merge_commit"] == merge


# --- 5. Journey 2's own refusals keep the rules of the other journeys.


def _the_approved_card_has_no_prefix(data: dict) -> None:
    cards_of(data)[1] = {"name": "Quarterly SENTINEL-NAME card"}


def _a_card_is_already_on_the_board() -> None:
    kanban_db.create_board(BOARD)
    with contextlib.closing(kanban_db.connect(board=BOARD)) as conn:
        kanban_db.create_task(conn, title="A card that was already there")


PREFIX_LINE = f'seed_states: a name in seed.json does not start with "{PREFIX}"; nothing was written\n'
JOURNEY_2_REFUSALS = {
    "card-name-without-the-prefix": (edited(_the_approved_card_has_no_prefix), None, PREFIX_LINE),
    "two-cards-only": (
        edited(lambda data: cards_of(data).pop()),
        None,
        "seed_states: seed.json does not have exactly three cards for this journey; nothing was written\n",
    ),
    "card-already-on-the-board": (
        SEED_TEXT,
        _a_card_is_already_on_the_board,
        "seed_states: the journey board already holds a card; nothing was written\n",
    ),
}


@pytest.mark.parametrize("case", sorted(JOURNEY_2_REFUSALS))
def test_journey_2_refuses_a_bad_seed_or_a_used_board_with_one_fixed_line_and_writes_nothing(
    seed, reset_home, capsys, case,
):
    text, set_up, line = JOURNEY_2_REFUSALS[case]
    if set_up:
        set_up()
    before = snapshot(reset_home)

    code, out, err = run_seed(seed, capsys, "2", text)

    assert (code, out, err) == (1, "", line)
    assert snapshot(reset_home) == before
    assert not (reset_home / "projects.db").exists()
    assert not (reset_home / LEDGER).exists()
