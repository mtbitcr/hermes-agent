"""Tests for the journeys seed (card 2): seed_states.py and seed_stack.sh.

seed_states.py runs for real against a temporary platform home: its commands are the real hermes
commands, its suggestion and give-up come from the real kernel, and the owner-workspace reads
(decisions list, board read) are the real ones. Those reads are made straight after the seed with
nothing played in between: the seed itself leaves the committed owner receipt and the board's owner
stamp that the reads need.
seed_stack.sh runs against a stand-in docker that sits first on PATH: it records every call, answers
inspect, and runs the snippets and the seed program that are sent into the dashboard on this machine.
No docker and no network is involved.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import importlib.util
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from agent import estop
from hermes_cli import kanban_db, owner_workspace as ow, projects_db, release_ledger
from hermes_cli.config import load_config
from plugins.dashboard_auth.raphael_workspace import model_policy

pytestmark = pytest.mark.linux_only

REPO = Path(__file__).resolve().parents[2]
JOURNEYS = REPO / "scripts" / "journeys"
SEED_STATES = JOURNEYS / "seed_states.py"
SEED_STACK = JOURNEYS / "seed_stack.sh"
TEMPLATE = JOURNEYS / "stack.env.template"
STACK_HOME = JOURNEYS / "stack-home"

BOARD = "journey-test"
PROJECT_SLUG = "journey-test-project"
PREFIX = "Journey test"
PROJECT_NAME = "Journey test project"
QUESTION = "Journey test question"
SUGGESTION = "Journey test suggestion"
STOPPED = "Journey test stopped card"

# SEED mirrors the structure that the review reported for the owner app's seed.json: journeys[n].projects
# arrays, each project with a name and nested questions, suggestions and cards. It is not a copy of that
# file, which is not on this machine, and any field the review did not report is not in it. Journey 2 carries
# three cards and a release batch: extra content that the seed for journeys 3 and 5 must leave alone.
SEED = {
    "prefix": PREFIX,
    "journeys": {
        "2": {
            "projects": [
                {
                    "name": PROJECT_NAME,
                    "questions": [],
                    "suggestions": [],
                    "cards": [
                        {"name": "Journey test card waiting for review"},
                        {"name": "Journey test card approved"},
                        {"name": "Journey test card merged"},
                    ],
                    "release_batch": {"name": "Journey test release batch", "cards": ["Journey test card merged"]},
                }
            ]
        },
        "3": {
            "projects": [
                {
                    "name": PROJECT_NAME,
                    "questions": [{"name": QUESTION}],
                    "suggestions": [{"name": SUGGESTION}],
                    "cards": [],
                }
            ]
        },
        "5": {
            "projects": [
                {
                    "name": PROJECT_NAME,
                    "questions": [],
                    "suggestions": [],
                    "cards": [{"name": STOPPED}],
                }
            ]
        },
    },
}
SEED_TEXT = json.dumps(SEED, indent=2) + "\n"

# Where the released owner app copy keeps seed.json inside the browser container.
SEED_FILE = "/opt/raphael-journeys/tools/page-check/journeys/seed.json"

# Bytecode of the platform modules that the seed's commands import is cached here, never in the tree.
REAL_PYC = Path(tempfile.gettempdir()) / f"journeys-seed-pyc-{os.getuid()}"  # windows-footgun: ok - the module is linux_only

# The docker --format shapes that seed_stack.sh asks for (the same ones reset_stack.sh uses).
F_STATE = '{{.State.Running}} {{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}'
F_DEST = '{{range .Mounts}}{{println .Name .Destination}}{{end}}'
F_HOME = '{{range .Config.Env}}{{$p := split . "="}}{{if eq (index $p 0) "HERMES_HOME"}}{{index $p 1}}{{end}}{{end}}'

MODEL_TOKEN = "sk-ant-oat01-SENTINEL-model-token-0123456789"


def template_names(section: str) -> list[str]:
    names, current = [], None
    for raw in TEMPLATE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        header = re.fullmatch(r"\[([^\]]+)\]", line)
        if header:
            current = header.group(1)
        elif current == section:
            names.append(line)
    return names


def tree(root: Path) -> dict[str, str]:
    """Relative path -> content hash, for proving a folder was not touched.

    Files whose names end in -wal or -shm are left out: they are SQLite's own sidecar files, which a read may create
    or touch, and owner rule 1 accepts that.
    """
    seen: dict[str, str] = {}
    for base, dirs, files in os.walk(root):
        for name in sorted(dirs) + sorted(f for f in files if not f.endswith(("-wal", "-shm"))):
            full = Path(base) / name
            seen[str(full.relative_to(root))] = "dir" if full.is_dir() else hashlib.sha256(full.read_bytes()).hexdigest()
    return seen


def rows_of(root: Path) -> dict[str, dict[str, list[tuple]]]:
    """File -> table -> every row, for each SQLite file under root, read through a read-only connection.

    A file is a SQLite file if it starts with SQLite's 16-byte header. Virtual tables are skipped, because their rows
    live in their own shadow tables. Rows are sorted by repr, so that the page order does not matter.
    """
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


def first(db: Path, sql: str):
    """The first column of the first row, read here and not by the seed."""
    with contextlib.closing(sqlite3.connect(f"file:{db}?mode=ro", uri=True)) as conn:
        return conn.execute(sql).fetchone()[0]


def count(db: Path, sql: str) -> int:
    return first(db, sql)


def edited(change) -> str:
    """SEED as JSON text after change(data) has altered a deep copy of it."""
    data = copy.deepcopy(SEED)
    change(data)
    return json.dumps(data)


def project_of(data: dict, journey: str) -> dict:
    return data["journeys"][journey]["projects"][0]


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


def run_seed(seed, capsys, journey: str, text: str = SEED_TEXT) -> tuple[int, str, str]:
    code = seed.main([journey], text)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def journey_3_decisions() -> list[tuple]:
    """What the owner workspace lists for the owner, read with the platform's own function."""
    decisions = ow.list_owner_decisions(ow.resolve_owner_context())["data"]
    return sorted((d["kind"], d["authority"], d["title"], d["project_name"]) for d in decisions)


JOURNEY_3_DECISIONS = [
    ("capability", "recommendation", SUGGESTION, PROJECT_NAME),
    ("owner_input", "task", QUESTION, PROJECT_NAME),
]


# --- 0. What "writes nothing" means (owner rule 1): the snapshot that the refusal tests compare.


def test_the_snapshot_holds_the_rows_and_the_store_files_and_leaves_the_sqlite_sidecars_out(tmp_path):
    """Owner rule 1: no row changes and no store file's own content changes; SQLite's own -wal and -shm may be touched."""
    home = tmp_path / "home"
    home.mkdir()
    db, config = home / "store.db", home / "config.yaml"
    wal, shm = db.with_name(db.name + "-wal"), db.with_name(db.name + "-shm")
    with contextlib.closing(sqlite3.connect(db)) as conn:
        assert conn.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
        conn.execute("CREATE TABLE cards (id INTEGER PRIMARY KEY, title TEXT)")
        conn.execute("INSERT INTO cards (title) VALUES ('first')")
        conn.commit()
    config.write_text("kanban:\n  dispatch_in_gateway: false\n", encoding="utf-8")
    assert not wal.exists() and not shm.exists()
    before = snapshot(home)
    main_file = db.read_bytes()

    # A second connection keeps the log open, so that SQLite itself creates and rewrites -wal and -shm.
    keeper = sqlite3.connect(db)
    try:
        keeper.execute("PRAGMA wal_autocheckpoint=0")
        # (a) The sidecars appear and are rewritten by two commits that leave every row as it was: the snapshot stays.
        keeper.execute("SELECT * FROM cards").fetchall()
        assert wal.exists() and shm.exists()
        assert snapshot(home) == before
        sidecars = (wal.read_bytes(), shm.read_bytes())
        for title in ("other", "first"):
            keeper.execute("UPDATE cards SET title = ?", (title,))
            keeper.commit()
        assert (wal.read_bytes(), shm.read_bytes()) != sidecars
        assert snapshot(home) == before

        # (b) A row changes. It is held only in the log, so the main file cannot show it: only the rows can.
        keeper.execute("UPDATE cards SET title = 'second'")
        keeper.commit()
        assert db.read_bytes() == main_file
        assert snapshot(home) != before
    finally:
        keeper.close()

    # (c) The content of an ordinary store file changes.
    before = snapshot(home)
    config.write_text("kanban:\n  dispatch_in_gateway: true\n", encoding="utf-8")
    assert snapshot(home) != before

    # (d) A new store file appears.
    before = snapshot(home)
    with contextlib.closing(sqlite3.connect(home / "projects.db")) as conn:
        conn.execute("CREATE TABLE projects (slug TEXT)")
        conn.commit()
    assert snapshot(home) != before


# --- 1. Journey 3: one waiting question and one waiting suggestion, no release record.


def test_journey_3_leaves_one_question_and_one_suggestion_waiting_for_the_owner(seed, reset_home, capsys):
    code, out, err = run_seed(seed, capsys, "3")
    assert code == 0, err

    # Read straight after the seed: exactly one question and one suggestion, both under the journey's Project.
    assert journey_3_decisions() == JOURNEY_3_DECISIONS
    assert not release_ledger.ledger_path().exists()


# --- 2. Journey 5: a card stopped after repeated failures, the dispatcher off, no worker.


def test_journey_5_leaves_a_stopped_card_with_the_dispatcher_off_and_starts_no_worker(
    seed, reset_home, capsys, monkeypatch,
):
    def a_worker_was_started(*args, **kwargs):
        raise AssertionError("the seed started a worker")

    monkeypatch.setattr(kanban_db, "_default_spawn", a_worker_was_started)
    code, out, err = run_seed(seed, capsys, "5")
    assert code == 0, err
    assert "seed_states: dispatcher setting: off\n" in out

    # Read straight after the seed: the Project is the owner's, and its board shows the one stopped card.
    ctx = ow.resolve_owner_context()
    snapshot = ow.read_project_snapshot(ctx, PROJECT_SLUG)
    shown = [(column["name"], task) for column in snapshot["columns"] for task in column["tasks"]]
    assert [task["title"] for _, task in shown] == [STOPPED]
    column, stopped = shown[0]
    assert column == "blocked"
    assert stopped["stopped_work"] == "gave_up"
    assert isinstance(stopped["event_revision"], int) and stopped["event_revision"] > 0

    with contextlib.closing(kanban_db.connect(board=BOARD)) as conn:
        card = kanban_db.get_task(conn, kanban_db.list_tasks(conn)[0].id)
        assert card.worker_pid is None and card.current_run_id is None

    # The dispatcher setting reads off in the copied config, and the seed left it as the reset wrote it.
    assert load_config()["kanban"]["dispatch_in_gateway"] is False
    assert (reset_home / "config.yaml").read_text(encoding="utf-8") == (STACK_HOME / "config.yaml").read_text(encoding="utf-8")

    # What "work dispatcher off" can mean: the owner-workspace answer carries one execution state, and it
    # moves with neither candidate. Not with the global pause, and not with the gateway setting.
    execution = snapshot["steward"]["execution"]
    estop.engage()
    assert estop.is_engaged()
    assert ow.read_project_snapshot(ctx, PROJECT_SLUG)["steward"]["execution"] == execution
    estop.disengage()
    config = reset_home / "config.yaml"
    config.write_text(config.read_text(encoding="utf-8").replace("dispatch_in_gateway: false", "dispatch_in_gateway: true"), encoding="utf-8")
    assert load_config()["kanban"]["dispatch_in_gateway"] is True
    assert ow.read_project_snapshot(ctx, PROJECT_SLUG)["steward"]["execution"] == execution


ROUTE_COLUMNS = ("assignee", "provider_override", "model_override", "reasoning_effort", "execution_tier", "model_policy_lock")


def route_of(card_id: str) -> dict:
    with contextlib.closing(kanban_db.connect(board=BOARD)) as conn:
        card = kanban_db.get_task(conn, card_id)
    return {column: getattr(card, column) for column in ROUTE_COLUMNS}


def test_journey_5_stopped_card_holds_the_route_that_make_review_card_gives(seed, reset_home, capsys):
    code, out, err = run_seed(seed, capsys, "5")
    assert code == 0, err
    with contextlib.closing(kanban_db.connect(board=BOARD)) as conn:
        (stopped,) = [task.id for task in kanban_db.list_tasks(conn)]

    # The reference is a card made by the seed's own make_review_card, on the same board.
    with seed.at_root(kanban_db.kanban_home()):
        reference = seed.make_review_card("Journey test route reference")

    route = route_of(stopped)
    assert route == route_of(reference)

    # Equal is not enough if both are empty: it is the route that REVIEW_ROUTE and the model policy admit,
    # and its lock validates for exactly those columns.
    assignee, provider, tier = seed.REVIEW_ROUTE
    admitted = model_policy.task_assignment_for(assignee, provider, tier)
    assert route == {
        "assignee": assignee,
        "provider_override": admitted.provider,
        "model_override": admitted.model,
        "reasoning_effort": admitted.reasoning_effort,
        "execution_tier": tier,
        "model_policy_lock": route["model_policy_lock"],
    }
    assert route["model_policy_lock"]
    assert kanban_db.policy_lock_error(
        route["model_policy_lock"], assignee, admitted.provider, admitted.model, admitted.reasoning_effort, tier
    ) is None
    assert stopped != reference


def receipt_rows(registry: Path) -> list[tuple]:
    with contextlib.closing(sqlite3.connect(f"file:{registry}?mode=ro", uri=True)) as conn:
        return conn.execute("SELECT * FROM owner_workspace_receipts ORDER BY idempotency_key").fetchall()


def test_the_seed_commits_one_owner_receipt_and_a_repeat_replays_it(seed, reset_home, capsys):
    code, out, err = run_seed(seed, capsys, "3")
    assert code == 0, err

    registry = reset_home / "projects.db"
    stamp = reset_home / "kanban" / "boards" / BOARD / "board.json"
    rows = receipt_rows(registry)
    assert len(rows) == 1
    project_id = first(registry, f"SELECT id FROM projects WHERE slug = '{PROJECT_SLUG}'")
    assert json.loads(stamp.read_text(encoding="utf-8"))["project_id"] == project_id
    stamped = stamp.read_bytes()

    # A repeat finds the same fixed key and digest: it replays, and nothing in the receipt or the stamp moves.
    seed.commit_owner_receipt()
    assert receipt_rows(registry) == rows
    assert stamp.read_bytes() == stamped
    assert journey_3_decisions() == JOURNEY_3_DECISIONS


def test_a_platform_without_a_receipt_step_is_refused_before_anything_is_written(seed, reset_home, capsys, monkeypatch):
    monkeypatch.delattr(ow, "_finalize_receipt")
    before = snapshot(reset_home)

    code, out, err = run_seed(seed, capsys, "3")

    assert (code, out) == (1, "")
    assert err == "seed_states: the platform lacks a step that commits the owner receipt; nothing was written\n"
    assert snapshot(reset_home) == before
    assert not (reset_home / "projects.db").exists()


# --- 3. Refusals: each prints one fixed line, exits 1 and writes nothing (owner rule 1).

PREFIX_LINE =f'seed_states: a name in seed.json does not start with "{PREFIX}"; nothing was written\n'


def _without_journey_5(text: str) -> str:
    data = json.loads(text)
    del data["journeys"]["5"]
    return json.dumps(data)


def _with_question(name: str) -> str:
    return edited(lambda data: project_of(data, "3").update(questions=[name]))


def _a_card_is_already_on_the_board() -> None:
    kanban_db.create_board(BOARD)
    with contextlib.closing(kanban_db.connect(board=BOARD)) as conn:
        kanban_db.create_task(conn, title="A card that was already there")


def _the_dispatcher_setting_is_on() -> None:
    config = Path(os.environ["HERMES_HOME"]) / "config.yaml"
    config.write_text(
        config.read_text(encoding="utf-8").replace("dispatch_in_gateway: false", "dispatch_in_gateway: true"), encoding="utf-8"
    )


REFUSALS = {
    "name-without-the-prefix": ("3", _with_question("Quarterly SENTINEL-NAME question"), None),
    "project-name-without-the-prefix": (
        "3", edited(lambda data: project_of(data, "3").update(name="Quarterly SENTINEL-NAME project")), None,
    ),
    "suggestion-name-without-the-prefix": (
        "3", edited(lambda data: project_of(data, "3").update(suggestions=["Quarterly SENTINEL-NAME suggestion"])), None,
    ),
    "stopped-card-name-without-the-prefix": (
        "5", edited(lambda data: project_of(data, "5").update(cards=[{"title": "Quarterly SENTINEL-NAME card"}])), None,
    ),
    "prefix-field-is-not-the-prefix": ("3", edited(lambda data: data.update(prefix="Quarterly SENTINEL-NAME")), None),
    "journey-seed-json-lacks": ("5", _without_journey_5(SEED_TEXT), None),
    "journey-4": ("4", SEED_TEXT, None),
    "card-already-on-the-board": ("3", SEED_TEXT, _a_card_is_already_on_the_board),
    "dispatcher-setting-not-off": ("5", SEED_TEXT, _the_dispatcher_setting_is_on),
}


@pytest.mark.parametrize("case", sorted(REFUSALS))
def test_the_seed_refuses_with_one_fixed_line_and_writes_nothing(seed, reset_home, capsys, case):
    journey, text, set_up = REFUSALS[case]
    if set_up:
        set_up()
    before = snapshot(reset_home)

    code, out, err = run_seed(seed, capsys, journey, text)

    assert code == 1
    assert out == ""
    assert len(err.strip().splitlines()) == 1
    assert "SENTINEL-NAME" not in err
    if "without-the-prefix" in case or case == "prefix-field-is-not-the-prefix":
        assert err == PREFIX_LINE
    # Nothing was written (owner rule 1): no row and no store file's content changed; no project, no new board, no new card.
    assert snapshot(reset_home) == before
    assert (reset_home / "kanban" / "boards" / BOARD).exists() is (case == "card-already-on-the-board")
    assert not (reset_home / "projects.db").exists()
    if case == "card-already-on-the-board":
        with contextlib.closing(kanban_db.connect(board=BOARD)) as conn:
            assert [task.title for task in kanban_db.list_tasks(conn)] == ["A card that was already there"]


# --- 3b. Fix: a board that holds a card of ANY kind or status is not a fresh board.


def _an_archived_work_card_is_already_on_the_board() -> None:
    kanban_db.create_board(BOARD)
    with contextlib.closing(kanban_db.connect(board=BOARD)) as conn:
        kanban_db.archive_task(conn, kanban_db.create_task(conn, title="A card that was already there"))
    db = kanban_db.kanban_db_path(BOARD)
    assert count(db, "SELECT COUNT(*) FROM tasks WHERE task_kind = 'work' AND status = 'archived'") == 1


def _only_a_suggestion_is_already_on_the_board() -> None:
    """A board that kanban_db.list_tasks reports as empty: it holds one native recommendation and no work card."""
    kanban_db.create_board(BOARD)
    with contextlib.closing(projects_db.connect()) as conn:
        project_id = projects_db.create_project(conn, name="An older project", slug="an-older-project", board_slug=BOARD)
    evidence = {
        "schema_version": 1,
        "need": "An older need.",
        "expected_benefit": "An older benefit.",
        "requested_scope": dict.fromkeys(kanban_db.RECOMMENDATION_SCOPE_FLAGS, False),
        "risks": "None.",
        "cost": "None.",
        "rollback": "Reject it.",
    }
    with contextlib.closing(kanban_db.connect(board=BOARD)) as conn:
        kanban_db.create_recommendation(
            conn,
            project_id=project_id,
            target_profile="default",
            recommendation_kind="skill",
            recommendation_subject_id="an-older-suggestion",
            recommendation_label="An older suggestion",
            recommendation_rationale="Waiting since before the seed.",
            recommendation_evidence=evidence,
            provenance_authority="hermes-profile:default",
        )
        assert kanban_db.list_tasks(conn, include_archived=True) == []
    db = kanban_db.kanban_db_path(BOARD)
    assert count(db, "SELECT COUNT(*) FROM tasks WHERE task_kind = 'work'") == 0
    assert count(db, "SELECT COUNT(*) FROM tasks WHERE task_kind = 'recommendation'") == 1


BOARD_HOLDS_A_CARD = {
    "work-card": _a_card_is_already_on_the_board,
    "archived-work-card": _an_archived_work_card_is_already_on_the_board,
    "suggestion-only": _only_a_suggestion_is_already_on_the_board,
}


@pytest.mark.parametrize("journey", ["3", "5"])
@pytest.mark.parametrize("case", sorted(BOARD_HOLDS_A_CARD))
def test_a_board_that_holds_a_card_of_any_kind_is_refused_and_the_home_is_left_as_it_was(
    seed, reset_home, capsys, case, journey,
):
    BOARD_HOLDS_A_CARD[case]()
    before = snapshot(reset_home)

    code, out, err = run_seed(seed, capsys, journey)

    assert (code, out) == (1, "")
    assert err == "seed_states: the journey board already holds a card; nothing was written\n"
    assert snapshot(reset_home) == before


def test_a_card_that_only_the_write_ahead_log_holds_is_still_seen_by_the_check(seed, reset_home, capsys):
    kanban_db.create_board(BOARD)
    db = kanban_db.kanban_db_path(BOARD)
    wal = db.with_name(db.name + "-wal")
    # The platform keeps a board out of WAL mode on a machine whose SQLite has the WAL-reset bug, so this writer
    # switches the board to WAL itself and holds its card in the log, as a running dashboard does on a newer SQLite.
    writer = sqlite3.connect(db)
    writer.row_factory = sqlite3.Row
    try:
        assert writer.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
        writer.execute("PRAGMA wal_autocheckpoint=0")
        kanban_db.create_task(writer, title="A card that is still in the log")
        with contextlib.closing(sqlite3.connect(db.resolve().as_uri() + "?mode=ro&immutable=1", uri=True)) as main_file:
            assert main_file.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 0
        assert wal.exists() and wal.stat().st_size > 0
        # Owner rule 1: the board file's bytes and the rows stay as they are; the read may touch -wal and -shm.
        held = (db.read_bytes(), rows_of(db.parent))

        code, out, err = run_seed(seed, capsys, "3")

        assert (db.read_bytes(), rows_of(db.parent)) == held
    finally:
        writer.close()
    assert (code, out) == (1, "")
    assert err == "seed_states: the journey board already holds a card; nothing was written\n"
    assert not (reset_home / "projects.db").exists()


# --- 3c. Fix: the parser reads the structure that the owner app's seed.json has.


def test_the_names_come_from_the_projects_arrays_of_the_journeys(seed):
    assert seed.read_names(SEED_TEXT, "3") == {"project": PROJECT_NAME, "question": QUESTION, "suggestion": SUGGESTION}
    assert seed.read_names(SEED_TEXT, "5") == {"project": PROJECT_NAME, "stopped": STOPPED}


def test_a_list_item_is_a_plain_string_or_an_object_named_by_name_or_title(seed):
    def as_strings_and_titles(data):
        project_of(data, "3")["questions"] = [QUESTION]
        project_of(data, "3")["suggestions"] = [{"title": SUGGESTION}]
        project_of(data, "5")["cards"] = [STOPPED]

    text = edited(as_strings_and_titles)

    assert seed.read_names(text, "3") == {"project": PROJECT_NAME, "question": QUESTION, "suggestion": SUGGESTION}
    assert seed.read_names(text, "5") == {"project": PROJECT_NAME, "stopped": STOPPED}


def test_journeys_may_be_a_list_of_objects_numbered_by_journey_number_or_id(seed):
    entries = SEED["journeys"]
    listed = [{"journey": 2, **entries["2"]}, {"number": "3", **entries["3"]}, {"id": 5, **entries["5"]}]
    text = json.dumps({"journeys": listed})

    assert seed.read_names(text, "3") == {"project": PROJECT_NAME, "question": QUESTION, "suggestion": SUGGESTION}
    assert seed.read_names(text, "5") == {"project": PROJECT_NAME, "stopped": STOPPED}


def test_a_list_is_found_below_the_project_when_the_project_has_no_such_child(seed):
    def move_down(data):
        project = project_of(data, "3")
        project["waiting"] = {
            "for_the_owner": [{"questions": project.pop("questions")}],
            "advice": {"suggestions": project.pop("suggestions")},
        }

    assert seed.read_names(edited(move_down), "3") == {"project": PROJECT_NAME, "question": QUESTION, "suggestion": SUGGESTION}


def test_a_direct_child_wins_over_a_list_of_the_same_key_further_down(seed):
    def add_a_copy_below(data):
        project_of(data, "3")["history"] = {"questions": ["Quarterly SENTINEL-NAME question"]}

    assert seed.read_names(edited(add_a_copy_below), "3") == {
        "project": PROJECT_NAME, "question": QUESTION, "suggestion": SUGGESTION,
    }


def test_other_journeys_and_keys_the_journey_does_not_need_are_ignored(seed):
    def add_noise(data):
        data["journeys"]["7"] = {"projects": "SENTINEL-NAME", "note": ["not", "a", "journey"]}
        data["journeys"]["4"] = {"projects": [{"name": "Quarterly SENTINEL-NAME project", "cards": [1, 2]}]}
        project_of(data, "3")["cards"] = [{"name": "Quarterly SENTINEL-NAME card"}, {"name": "Another"}]
        project_of(data, "5")["questions"] = [1, 2, 3]
        project_of(data, "5")["suggestions"] = "SENTINEL-NAME"
        data["unrelated"] = {"cards": 7}

    text = edited(add_noise)

    assert seed.read_names(text, "3") == {"project": PROJECT_NAME, "question": QUESTION, "suggestion": SUGGESTION}
    assert seed.read_names(text, "5") == {"project": PROJECT_NAME, "stopped": STOPPED}


def marked(text: str) -> str:
    """Every name that the parser reads carries a marker, so that a line which echoed a value would show it."""
    for name in (PROJECT_NAME, QUESTION, SUGGESTION, STOPPED):
        text = text.replace(name, f"{name} SENTINEL-NAME")
    return text


def _two_lists_below(key: str, value):
    def change(data):
        project = project_of(data, "3")
        project.pop(key)
        project["left"] = {key: value}
        project["right"] = [{key: value}]

    return edited(change)


# case: (journey, seed.json text, words that the one line must use to name the missing part)
SHAPE_REFUSALS = {
    "text-is-not-json": ("3", '{"journeys": SENTINEL-NAME', "JSON object"),
    "text-is-not-an-object": ("3", '["SENTINEL-NAME"]', "JSON object"),
    "no-journeys": ("3", edited(lambda data: data.pop("journeys")), "no journeys"),
    "journey-not-listed": ("5", edited(lambda data: data["journeys"].pop("5")), "no entry for this journey"),
    "journey-listed-twice": (
        "3", json.dumps({"journeys": [{"journey": 3, **SEED["journeys"]["3"]}, {"number": "3", **SEED["journeys"]["3"]}]}),
        "more than one entry",
    ),
    "no-projects-list": ("3", edited(lambda data: data["journeys"]["3"].pop("projects")), "no projects list"),
    "projects-is-not-a-list": ("3", edited(lambda data: data["journeys"]["3"].update(projects={"name": "x"})), "no projects list"),
    "no-project": ("3", edited(lambda data: data["journeys"]["3"].update(projects=[])), "exactly one project"),
    "two-projects": ("3", edited(lambda data: data["journeys"]["3"]["projects"].append(project_of(data, "3"))), "exactly one project"),
    "project-has-no-name": ("3", edited(lambda data: project_of(data, "3").pop("name")), "project in seed.json has no name"),
    "no-questions-list": ("3", edited(lambda data: project_of(data, "3").pop("questions")), "no questions list"),
    "no-suggestions-list": ("3", edited(lambda data: project_of(data, "3").pop("suggestions")), "no suggestions list"),
    "no-cards-list": ("5", edited(lambda data: project_of(data, "5").pop("cards")), "no cards list"),
    "no-question": ("3", edited(lambda data: project_of(data, "3").update(questions=[])), "exactly one question"),
    "two-questions": (
        "3", edited(lambda data: project_of(data, "3")["questions"].append("Journey test second question")), "exactly one question",
    ),
    "two-suggestions": (
        "3", edited(lambda data: project_of(data, "3")["suggestions"].append({"name": "Journey test second suggestion"})),
        "exactly one suggestion",
    ),
    "two-cards": ("5", edited(lambda data: project_of(data, "5")["cards"].append(STOPPED)), "exactly one card"),
    "two-lists-of-a-key-below-the-project": ("3", _two_lists_below("questions", [QUESTION]), "more than one questions list"),
    "item-without-a-name": ("3", edited(lambda data: project_of(data, "3").update(questions=[{"text": "x"}])), "question in seed.json has no name"),
    "item-is-a-number": ("5", edited(lambda data: project_of(data, "5").update(cards=[7])), "card in seed.json has no name"),
}


@pytest.mark.parametrize("case", sorted(SHAPE_REFUSALS))
def test_a_seed_json_that_lacks_a_part_is_refused_with_one_line_naming_the_part(seed, reset_home, capsys, case):
    journey, text, part = SHAPE_REFUSALS[case]
    before = snapshot(reset_home)

    code, out, err = run_seed(seed, capsys, journey, marked(text))

    assert code == 1
    assert out == ""
    assert len(err.strip().splitlines()) == 1
    assert part in err
    assert err.startswith("seed_states: ") and err.endswith("; nothing was written\n")
    assert "SENTINEL-NAME" not in err
    assert snapshot(reset_home) == before
    assert not (reset_home / "projects.db").exists()
    assert not (reset_home / "kanban" / "boards" / BOARD).exists()


# --- 4. seed_stack.sh: the dashboard container, the home owner's ids, nothing secret.

HOME_OWNER = "10001:10002"

FAKE_DOCKER = r'''#!PYTHON_PATH
import json, os, subprocess, sys

state = json.load(open(os.environ["FAKE_DOCKER_STATE"]))
argv = sys.argv[1:]


def record(stdin=None):
    with open(state["log"], "a") as fh:
        fh.write(json.dumps({"argv": argv, "stdin": stdin}) + "\n")


def refuse(why):
    sys.stderr.write("fake docker: " + why + "\n")
    sys.exit(98)


def show(done):
    sys.stdout.write(done.stdout)
    sys.stderr.write(done.stderr)
    sys.exit(done.returncode)


if argv[:1] == ["inspect"]:
    record()
    if argv[1:4] != ["--type", "container", "--format"] or len(argv) != 6:
        refuse("inspect needs --type container --format FORMAT NAME")
    fmt, name = argv[4], argv[5]
    entry = state["containers"].get(name)
    if entry is None:
        sys.stderr.write("Error: No such container: " + name + "\n")
        sys.exit(1)
    if fmt == state["fmt_state"]:
        print("true none" if entry["running"] else "false none")
    elif fmt == state["fmt_dest"]:
        for mount in entry["mounts"]:
            print(mount["name"], mount["dest"])
    elif fmt == state["fmt_home"]:
        print(entry["home"])
    else:
        refuse("format not supported by the stand-in")
    sys.exit(0)

if argv[:1] != ["exec"]:
    record()
    refuse("only inspect and exec are allowed")

args, interactive = argv[1:], False
while args and args[0].startswith("-"):
    if args[0] == "-i":
        interactive, args = True, args[1:]
    elif args[0] == "--user":
        args = args[2:]
    else:
        record()
        refuse("exec option not supported by the stand-in")
container, cmd = args[0], args[1:]
data = sys.stdin.read() if interactive else None
record(data)
entry = state["containers"].get(container)
if entry is None:
    refuse("no such container")
if container == state["app"] and cmd in (["id", "-u"], ["id", "-g"]):
    print(state["app_user"].split(":")[0 if cmd[1] == "-u" else 1])
elif container == state["dashboard"] and cmd[:3] == ["stat", "-c", "%u:%g"]:
    print(state["home_owner"])
elif container == state["browser"] and len(cmd) == 2 and cmd[0] == "cat" and cmd[1] in state["files"]:
    sys.stdout.write(state["files"][cmd[1]])
elif container == state["dashboard"] and cmd[:2] == ["sh", "-c"]:
    show(subprocess.run(["sh", "-c", *cmd[2:]], env={"PATH": state["container_path"]}, capture_output=True, text=True, encoding="utf-8"))
elif container == state["dashboard"] and cmd[1:2] == ["-c"]:
    env = {
        "PATH": "/usr/bin:/bin", "HOME": state["scratch"], "HERMES_HOME": entry["host_home"],
        "PYTHONPATH": state["repo"], "PYTHONPYCACHEPREFIX": state["pyc"],
    }
    show(subprocess.run(cmd, input=data, env=env, capture_output=True, text=True, encoding="utf-8", cwd=state["repo"]))
else:
    refuse("command not supported by the stand-in")
'''


def split_exec(argv: list[str]) -> dict:
    args, user, interactive = argv[1:], None, False
    while args[0].startswith("-"):
        if args[0] == "-i":
            interactive, args = True, args[1:]
        else:
            assert args[0] == "--user", args
            user, args = args[1], args[2:]
    return {"user": user, "interactive": interactive, "container": args[0], "cmd": args[1:]}


def stand_in_stack(base: Path, app_user: str) -> tuple[dict, Path, Path]:
    """Five stand-in containers: the dashboard's home is a folder on this machine; docker and hostname are stand-ins."""
    proxy, app, dash, gate, browser = template_names("containers")
    base.mkdir(parents=True)
    platform_home = base / "dashboard-home"
    shutil.copytree(STACK_HOME, platform_home)
    (platform_home / ".env").write_text(f"API_SERVER_KEY={MODEL_TOKEN}\n", encoding="utf-8")
    bin_dir = base / "bin"
    bin_dir.mkdir()
    # The dashboard's PATH: the image's privilege-drop shim first, then the venv launcher, whose first line
    # names the platform's Python.
    shim = base / "container" / "opt" / "hermes" / "bin"
    venv = base / "container" / "opt" / "hermes" / ".venv" / "bin"
    shim.mkdir(parents=True)
    venv.mkdir(parents=True)
    (shim / "hermes").write_text('#!/bin/sh\nexec /opt/hermes/.venv/bin/hermes "$@"\n', encoding="utf-8")
    (venv / "hermes").write_text(f"#!{sys.executable}\nraise SystemExit(0)\n", encoding="utf-8")
    log = base / "calls.jsonl"
    state = {
        "log": str(log), "fmt_state": F_STATE, "fmt_dest": F_DEST, "fmt_home": F_HOME,
        "app": app, "dashboard": dash, "browser": browser, "home_owner": HOME_OWNER, "app_user": app_user,
        "container_path": f"{shim}:{venv}:/usr/bin:/bin", "scratch": str(base), "repo": str(REPO),
        "pyc": str(REAL_PYC), "files": {SEED_FILE: SEED_TEXT},
        "containers": {
            name: {"running": True, "mounts": [], "home": "", "host_home": ""} for name in (proxy, app, dash, gate, browser)
        },
    }
    state["containers"][dash].update(
        mounts=[{"name": template_names("volumes")[0], "dest": "/opt/data"}], home="/opt/data", host_home=str(platform_home)
    )
    state_file = base / "state.json"
    state_file.write_text(json.dumps(state), encoding="utf-8")
    (bin_dir / "docker").write_text(FAKE_DOCKER.replace("PYTHON_PATH", sys.executable), encoding="utf-8")
    (bin_dir / "hostname").write_text("#!/bin/sh\necho journeys-host.invalid\n", encoding="utf-8")
    for name in ("docker", "hostname"):
        (bin_dir / name).chmod(0o755)
    env = {
        "PATH": f"{bin_dir}:{os.environ['PATH']}", "HOME": str(base), "FAKE_DOCKER_STATE": str(state_file),
        "JOURNEY_STACK_HOST": "journeys-host.invalid", "CLAUDE_CODE_OAUTH_TOKEN": MODEL_TOKEN, "API_SERVER_KEY": MODEL_TOKEN,
    }
    return env, log, platform_home


def run_stack_script(env: dict, base: Path, journey: str) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(SEED_STACK), journey], env=env, cwd=base, capture_output=True, text=True, encoding="utf-8", timeout=240)


def test_seed_stack_runs_in_the_dashboard_as_the_home_owner_and_keeps_secrets_off_command_lines(tmp_path):
    proxy, app, dash, gate, browser = template_names("containers")
    env, log, platform_home = stand_in_stack(tmp_path / "stack", app_user="1000:1000")

    done = run_stack_script(env, tmp_path, "5")

    assert done.returncode == 0, done.stderr
    calls = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert {call["argv"][0] for call in calls} <= {"inspect", "exec"}
    execs = [split_exec(call["argv"]) | {"stdin": call["stdin"]} for call in calls if call["argv"][0] == "exec"]

    # The seed program: in the dashboard (the third entry of [containers]), as the home owner, with seed.json's
    # text on standard input and the journey number as its one argument.
    program = [e for e in execs if e["cmd"][:1] == [sys.executable]]
    assert len(program) == 1
    assert program[0]["container"] == dash
    assert program[0]["interactive"] is True
    assert program[0]["cmd"][0] == sys.executable and program[0]["cmd"][-1] == "5"
    assert program[0]["stdin"].strip() == SEED_TEXT.strip()

    # The only commands without the home owner's ids are the lookups of the ids themselves. Every other command
    # runs as the account that owns the dashboard's home folder: never root, never the owner app's user.
    lookups = {(e["container"], tuple(e["cmd"])) for e in execs if e["user"] is None}
    assert lookups == {(app, ("id", "-u")), (dash, ("stat", "-c", "%u:%g", "/opt/data"))}
    assert all(e["user"] == HOME_OWNER for e in execs if e["user"] is not None)
    assert {e["container"] for e in execs} == {app, dash, browser}
    assert [e["cmd"] for e in execs if e["container"] == browser] == [["cat", SEED_FILE]]
    assert not any(name in json.dumps(calls) for name in (proxy, gate))

    # Nothing secret is on a command line, on standard input or in the output.
    assert MODEL_TOKEN not in json.dumps(calls) + done.stdout + done.stderr

    # The program really ran in the home it was sent to: the stopped card is there.
    assert count(platform_home / "kanban" / "boards" / BOARD / "kanban.db", "SELECT COUNT(*) FROM tasks") == 1

    # A home folder that belongs to the owner app's own user is refused, and no program is started.
    env2, log2, home2 = stand_in_stack(tmp_path / "same-user", app_user=HOME_OWNER)
    refused = run_stack_script(env2, tmp_path, "5")
    assert refused.returncode == 1
    assert len(refused.stderr.strip().splitlines()) == 1
    calls2 = [json.loads(line) for line in log2.read_text(encoding="utf-8").splitlines()]
    assert not [c for c in calls2 if c["argv"][0] == "exec" and split_exec(c["argv"])["cmd"][:1] == [sys.executable]]
    assert not (home2 / "kanban" / "boards" / BOARD).exists()


# --- 5. The store paths: run from inside a worker, the seed still reaches the root's stores.


def test_the_seed_reaches_the_journey_board_and_the_root_stores_from_inside_a_worker(tmp_path, reset_home):
    root = reset_home
    kanban_db.create_board("worker-board")
    with contextlib.closing(kanban_db.connect(board="worker-board")) as conn:
        worker_task = kanban_db.create_task(conn, title="The worker's own card")
    with contextlib.closing(projects_db.connect()) as conn:
        projects_db.create_project(conn, name="An older project", slug="an-older-project")
    with contextlib.closing(release_ledger.connect()):
        pass
    profile_home = root / "profiles" / "raphael-executor"
    profile_home.mkdir(parents=True)

    env = {
        "PATH": os.environ["PATH"], "HOME": str(tmp_path), "PYTHONPATH": str(REPO), "PYTHONPYCACHEPREFIX": str(REAL_PYC),
        "HERMES_HOME": str(profile_home), "HERMES_PROFILE": "raphael-executor",
        "HERMES_KANBAN_DB": str(root / "kanban" / "boards" / "worker-board" / "kanban.db"),
        "HERMES_KANBAN_BOARD": "worker-board", "HERMES_KANBAN_TASK": worker_task,
        "HERMES_KANBAN_WORKSPACES_ROOT": str(root / "kanban" / "boards" / "worker-board" / "workspaces"),
        "HERMES_KANBAN_WORKSPACE": str(tmp_path), "HERMES_KANBAN_RUN_ID": "1", "HERMES_KANBAN_CLAIM_LOCK": "host:1",
    }
    done = subprocess.run(
        [sys.executable, str(SEED_STATES), "3"], input=SEED_TEXT, env=env, cwd=tmp_path,
        capture_output=True, text=True, encoding="utf-8", timeout=240,
    )
    assert done.returncode == 0, done.stderr

    reported = dict(re.findall(r"^seed_states: (.+): (\S+)$", done.stdout, re.M))
    journey_db = root / "kanban" / "boards" / BOARD / "kanban.db"
    registry = root / "projects.db"
    worker_db = root / "kanban" / "boards" / "worker-board" / "kanban.db"

    # What the seed reported against what the stores hold, counted here and not by the seed.
    assert int(reported["work cards on the journey board"]) == count(journey_db, "SELECT COUNT(*) FROM tasks WHERE task_kind = 'work'") == 1
    assert int(reported["projects in the root registry"]) == count(registry, "SELECT COUNT(*) FROM projects") == 2
    suggestions = count(journey_db, f"SELECT COUNT(*) FROM tasks WHERE task_kind = 'recommendation' AND id = '{reported['suggestion']}'")
    assert suggestions == 1
    assert count(journey_db, "SELECT COUNT(*) FROM tasks") == 2
    assert reported["release record"] == "present"
    assert (root / "kanban" / "release_ledger.db").exists()

    # The committed owner receipt is in the root registry, counted here, and the board carries its owner's stamp.
    project_id = first(registry, f"SELECT id FROM projects WHERE slug = '{PROJECT_SLUG}'")
    assert count(registry, "SELECT COUNT(*) FROM owner_workspace_receipts") == 1
    committed = (
        "SELECT COUNT(*) FROM owner_workspace_receipts WHERE status = 'committed' AND operation = 'owner_task_graph_commit' "
        f"AND project_id = '{project_id}' AND board_slug = '{BOARD}'"
    )
    assert count(registry, committed) == 1
    assert json.loads((root / "kanban" / "boards" / BOARD / "board.json").read_text(encoding="utf-8"))["project_id"] == project_id

    # The same immediate reads as without a worker: the owner sees the one question and the one suggestion.
    assert journey_3_decisions() == JOURNEY_3_DECISIONS

    # Nothing went to the worker's own board or to its profile.
    assert count(worker_db, "SELECT COUNT(*) FROM tasks") == 1
    assert not (profile_home / "projects.db").exists()
    assert not (profile_home / "kanban").exists()
