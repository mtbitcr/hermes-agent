"""design_compare designer_kit: the designer, on its own card and outside a review run, receives the capture
kit, the judge and the design-offer scripts on that card, byte for byte with one SHA-256 each (capture kit
plan, card A).

Every case runs under the environment the dispatcher gives a worker: ``HERMES_KANBAN_DB`` and
``HERMES_KANBAN_BOARD`` pinned to the worker's own board, ``HERMES_KANBAN_TASK``, ``HERMES_KANBAN_RUN_ID``
and ``HERMES_KANBAN_CLAIM_LOCK`` set, and ``HERMES_HOME`` the designer's profile home under the root. Every
call goes through the registry. What landed where is read independently of the tool's result: every
attachment row of every board's own file under the root, straight from SQLite, with the stored bytes.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import sqlite3
from pathlib import Path

import pytest

from tools import design_compare_tool  # noqa: F401  (registers the tool)
from tools.design_access import (
    BOARD_NOT_LISTED, BUILDER, DESIGNER, NOT_CURRENT_RUN, NOT_IN_TASK, REVIEWER, check_design_access,
)
from tools.registry import registry

REPO = Path(__file__).resolve().parents[2]
KIT_DIR = REPO / "scripts" / "design_compare"
JUDGE = REPO / "tools" / "design_compare_judge.py"
SCRIPTS = REPO / "optional-skills" / "creative" / "design-offer" / "scripts"
KIT = hashlib.sha256((KIT_DIR / "capture.mjs").read_bytes() + (KIT_DIR / "probe.js").read_bytes()).hexdigest()
ENV = ("HERMES_KANBAN_DB", "HERMES_KANBAN_BOARD", "HERMES_KANBAN_TASK", "HERMES_KANBAN_RUN_ID",
       "HERMES_KANBAN_CLAIM_LOCK")
REFUSED = "design compare refused: "
NOT_DESIGNER = "only the designer may use designer_kit, on its own card, outside a review run"
NOT_REVIEWER = "only the reviewer may use it, in a review run"
TOO_LARGE = "the files to deliver exceed 1 MiB in total"
MIB = 1024 * 1024


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sources(scripts: Path = SCRIPTS) -> list:
    """The delivered set of the plan, from the repository: the capture kit, the judge, then every regular
    file directly in the skill's scripts folder with a .py or .mjs script name, by name."""
    names = sorted(path.name for path in scripts.iterdir() if path.is_file() and not path.is_symlink()
                   and re.fullmatch(r"[a-z0-9_]+\.(py|mjs)", path.name))
    return [KIT_DIR / "capture.mjs", KIT_DIR / "probe.js", JUDGE, *(scripts / name for name in names)]


def write_config(**design) -> None:
    """``config.yaml`` in the worker's own HERMES_HOME (JSON is YAML)."""
    Path(os.environ["HERMES_HOME"], "config.yaml").write_text(json.dumps({"design_browser": design}),
                                                              encoding="utf-8")


def board_files(root: Path) -> dict:
    """Every board's own file found on disk under the root: the default board's at the root, the others
    under kanban/boards."""
    found = {path.parent.name: path for path in (root / "kanban" / "boards").glob("*/kanban.db")}
    return {"default": root / "kanban.db", **found} if (root / "kanban.db").is_file() else found


def census(root: Path) -> list:
    """Every attachment row of every board under the root: (board, card, attachment id, file name, uploader,
    run of its ``attached`` receipt, SHA-256 of the stored bytes)."""
    rows = []
    for board, path in sorted(board_files(root).items()):
        with contextlib.closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as conn:
            rows += [(board, task, att, name, by, run, sha(Path(stored).read_bytes())) for task, att, name, by, stored, run
                     in conn.execute(
                         "SELECT a.task_id, a.id, a.filename, a.uploaded_by, a.stored_path, (SELECT e.run_id FROM "
                         "task_events e WHERE e.task_id = a.task_id AND e.kind = 'attached' AND "
                         "json_extract(e.payload, '$.attachment_id') = a.id) FROM task_attachments a ORDER BY a.id")]
    return rows


class Kanban:
    """Three real boards under the root, each with another designer card in its own run, and the
    dispatcher's environment for the caller's card on ``web``."""

    def __init__(self, kb, root: Path, monkeypatch) -> None:
        self.kb, self.root, self.mp = kb, root, monkeypatch

    def start(self, assignee: str = DESIGNER, *, review_by: str | None = None, branch: str | None = None) -> dict:
        """The caller's card assigned to ``assignee`` and claimed (from the review lane after
        ``request_review`` when ``review_by`` names who reviews), then its worker's environment."""
        kb = self.kb
        for var in ENV:  # the boards are reached by their own files, as before a worker starts
            self.mp.delenv(var, raising=False)
        with contextlib.closing(kb.connect(board="web")) as conn:
            card = kb.create_task(conn, title="Design options", assignee=assignee)
            first = kb.claim_task(conn, card)
            if review_by:
                assert kb.request_review(conn, card, summary="ready", reviewer=review_by,
                                         expected_run_id=first.current_run_id)
                assert kb.claim_review_task(conn, card, claimer=f"{review_by}:1") is not None
            if branch:
                kb.set_branch_name(conn, card, branch.format(id=card))
            task = kb.get_task(conn, card)
        for key, value in {
            "HERMES_KANBAN_DB": str(board_files(self.root)["web"]), "HERMES_KANBAN_BOARD": "web",
            "HERMES_KANBAN_TASK": card, "HERMES_KANBAN_RUN_ID": str(task.current_run_id),
            "HERMES_KANBAN_CLAIM_LOCK": task.claim_lock or "",
        }.items():
            self.mp.setenv(key, value)
        return {"task": card, "run": task.current_run_id}

    def raw(self, sql: str, *params):
        """A separate read-only connection to the web board's own file."""
        with contextlib.closing(sqlite3.connect(board_files(self.root)["web"].as_uri() + "?mode=ro", uri=True)) as conn:
            return conn.execute(sql, params).fetchall()


@pytest.fixture
def kanban(tmp_path, monkeypatch):
    from hermes_cli import kanban_db as kb

    root = tmp_path / "root"
    home = root / "profiles" / DESIGNER
    home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    for var in ("HERMES_KANBAN_HOME", "HERMES_KANBAN_ATTACHMENTS_ROOT", "HERMES_DELEGATED_CHILD_CONTEXT", *ENV):
        monkeypatch.delenv(var, raising=False)
    kb.init_db(board=kb.DEFAULT_BOARD)
    for board in ("web", "elsewhere"):
        kb.create_board(board)
    for board in (kb.DEFAULT_BOARD, "web", "elsewhere"):
        with contextlib.closing(kb.connect(board=board)) as conn:
            kb.claim_task(conn, kb.create_task(conn, title="Another design", assignee=DESIGNER))
    write_config(boards=["web", "elsewhere"])
    return Kanban(kb, root, monkeypatch)


def call(args: dict):
    return registry.dispatch("design_compare", dict(args), task_id="design-compare-test")


def ok(result) -> dict:
    assert isinstance(result, str), result
    payload = json.loads(result)
    assert "error" not in payload, payload
    return payload


def error(result) -> str:
    assert isinstance(result, str), result
    return json.loads(result)["error"]


# ── Delivery ─────────────────────────────────────────────────────────────────


def test_the_designer_on_its_own_card_receives_every_file_byte_for_byte_and_nothing_lands_elsewhere(kanban):
    ctx = kanban.start()
    assert set(board_files(kanban.root)) == {b["slug"] for b in kanban.kb.list_boards()} == {"default", "web",
                                                                                              "elsewhere"}
    expected = sources()
    assert len(expected) > 3  # the skill's scripts folder holds at least one script
    result = ok(call({"action": "designer_kit"}))
    assert result["action"] == "designer_kit"
    assert [entry["filename"] for entry in result["files"]] == [path.name for path in expected]
    for entry, path in zip(result["files"], expected):
        data = path.read_bytes()
        assert (entry["sha256"], entry["bytes"]) == (sha(data), len(data)), path
    assert result["kit_sha256"] == KIT
    assert census(kanban.root) == [("web", ctx["task"], entry["attachment_id"], entry["filename"], "design_compare",
                                    ctx["run"], entry["sha256"]) for entry in result["files"]]


def test_only_regular_script_files_directly_in_the_scripts_folder_are_delivered(kanban, tmp_path, monkeypatch):
    scripts, outside = tmp_path / "scripts", tmp_path / "outside.py"
    (scripts / "helpers").mkdir(parents=True)
    (scripts / "helpers" / "inner.py").write_bytes(b"INNER = 'one folder down'\n")
    (scripts / "nested.py").mkdir()  # a folder with a script's name
    outside.write_bytes(b"SECRET = 'not in the scripts folder'\n")
    (scripts / "linked.py").symlink_to(outside)
    (scripts / "compose_offer.py").write_bytes(b"print('compose')\n")
    (scripts / "a_first.mjs").write_bytes(b"export const first = 1;\n")
    for name in ("README.md", "Draft.py", "compose-offer.py"):
        (scripts / name).write_bytes(b"not a script name\n")
    monkeypatch.setattr("tools.design_compare_kit.SCRIPTS_DIR", scripts)
    ctx = kanban.start()
    result = ok(call({"action": "designer_kit"}))
    expected = [*sources(SCRIPTS)[:3], scripts / "a_first.mjs", scripts / "compose_offer.py"]
    assert [entry["filename"] for entry in result["files"]] == [path.name for path in expected]
    assert [(board, task, name, digest) for board, task, _, name, _, _, digest in census(kanban.root)] == [
        ("web", ctx["task"], path.name, sha(path.read_bytes())) for path in expected]


def test_up_to_one_mib_in_total_is_delivered_and_one_byte_more_is_refused_storing_nothing(kanban, tmp_path,
                                                                                         monkeypatch):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    kit = sum(path.stat().st_size for path in sources(SCRIPTS)[:3])
    (scripts / "large.py").write_bytes(b"#" * (MIB - kit))
    monkeypatch.setattr("tools.design_compare_kit.SCRIPTS_DIR", scripts)
    kanban.start()
    assert sum(entry["bytes"] for entry in ok(call({"action": "designer_kit"}))["files"]) == MIB
    before = census(kanban.root)
    assert len(before) == 4
    (scripts / "more.py").write_bytes(b"#")
    assert error(call({"action": "designer_kit"})) == REFUSED + TOO_LARGE
    assert census(kanban.root) == before


# ── Access: the designer alone, on its own card, outside a review run ────────


@pytest.mark.parametrize("case", ["designer-in-a-review-run", "reviewer", "builder", "designer-run-on-a-builder-card"])
def test_refused_to_everyone_but_the_designer_on_its_own_card_outside_a_review_run(kanban, case):
    config = {"boards": ["web"]}
    if case == "designer-in-a-review-run":  # a review asked of the designer: the card is the designer's now
        ctx, profile, assignee = kanban.start(BUILDER, review_by=DESIGNER), DESIGNER, DESIGNER
        assert kanban.raw("SELECT json_extract(payload, '$.source_status') FROM task_events WHERE run_id = ? "
                          "AND kind = 'claimed'", ctx["run"]) == [("review",)]
    elif case == "reviewer":
        ctx, profile, assignee = kanban.start(BUILDER, review_by=REVIEWER), REVIEWER, REVIEWER
    elif case == "builder":  # a builder run the gate lets in: its own preview exists
        ctx, profile, assignee = kanban.start(BUILDER, branch="web/{id}"), BUILDER, BUILDER
        config["allowed_hosts"] = ["*.raphael-first-project.pages.dev"]
    else:
        ctx, profile, assignee = kanban.start(), DESIGNER, BUILDER
        with contextlib.closing(sqlite3.connect(board_files(kanban.root)["web"])) as conn:
            conn.execute("UPDATE tasks SET assignee = ? WHERE id = ?", (BUILDER, ctx["task"]))
            conn.commit()
    write_config(**config)
    assert kanban.raw("SELECT r.profile, t.assignee FROM task_runs r JOIN tasks t ON t.id = r.task_id "
                      "WHERE r.id = ?", ctx["run"]) == [(profile, assignee)]
    access = check_design_access(config)
    assert (access.allowed, access.role) == (True, profile)  # the shared gate lets it through
    assert error(call({"action": "designer_kit"})) == REFUSED + NOT_DESIGNER
    assert census(kanban.root) == []


@pytest.mark.parametrize("case", ["stale-run", "unlisted-board", "no-kanban-task"])
def test_the_shared_gate_refuses_a_stale_run_an_unlisted_board_and_no_task_storing_nothing(kanban, monkeypatch,
                                                                                           case):
    ctx = kanban.start()
    ok(call({"action": "designer_kit"}))
    before = census(kanban.root)
    if case == "stale-run":  # the run ends and the card's next run starts; this worker's run id is stale
        with contextlib.closing(kanban.kb.connect(board="web")) as conn:
            assert kanban.kb.reclaim_task(conn, ctx["task"], reason="stale run")
            assert kanban.kb.claim_task(conn, ctx["task"]).current_run_id != ctx["run"]
        reason = NOT_CURRENT_RUN
    elif case == "unlisted-board":
        write_config(boards=["elsewhere"])
        reason = BOARD_NOT_LISTED
    else:
        monkeypatch.delenv("HERMES_KANBAN_TASK")
        reason = NOT_IN_TASK
    assert error(call({"action": "designer_kit"})) == REFUSED + reason
    assert census(kanban.root) == before


def test_the_designer_who_receives_the_kit_is_still_refused_prepare_judge_and_classify(kanban):
    kanban.start()
    ok(call({"action": "designer_kit"}))
    before = census(kanban.root)
    for args in ({"action": "prepare"}, {"action": "judge", "captures": []},
                 {"action": "classify", "regions": [{"id": "0" * 16, "category": "none", "reason": "same."}]}):
        assert error(call(args)) == REFUSED + NOT_REVIEWER
    assert census(kanban.root) == before
