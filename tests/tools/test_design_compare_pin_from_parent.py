"""design_compare finds the chosen full-page design on the designer card (design comparison own-route plan,
question 7, card 1).

A build card whose body holds no ``Prototype pin:`` line is compared with the ``design_pin`` record that the
one direct parent carrying the owner's ``design_option_chosen`` event wrote in its latest completed run. Every
case runs under the environment the dispatcher gives a worker, and every call goes through the registry.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import sqlite3
from pathlib import Path

import pytest
from PIL import Image

from hermes_cli.owner_design_choices import accept_choice, declare_design_options
from tools import design_compare_tool  # noqa: F401  (registers the tool)
from tools.design_access import BUILDER, DESIGNER, REVIEWER
from tools.registry import registry

ENV = ("HERMES_KANBAN_DB", "HERMES_KANBAN_WORKSPACES_ROOT", "HERMES_KANBAN_BOARD", "HERMES_KANBAN_TASK",
       "HERMES_KANBAN_RUN_ID", "HERMES_KANBAN_CLAIM_LOCK", "HERMES_PROFILE")
TITLE = "Privacy page"
PAGE = b"<!doctype html><title>chosen</title><h1>Privacy</h1>\n"
OTHER = b"<!doctype html><title>not chosen</title><h1>Privacy</h1>\n"
NO_PIN = "this card names no chosen design file"
NO_RECORD = "the chosen design has no full-page design file on its designer card"
STOP = {"verdict": "stop", "question": f"The chosen design file for '{TITLE}' is not the one this card was "
                                       "created with. Which design should Raphael compare against?"}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def picture() -> bytes:
    out = io.BytesIO()
    Image.new("RGB", (4, 4), "white").save(out, "PNG")
    return out.getvalue()


class Kanban:
    """Real boards under the root, build cards with their designer cards, and each build card's review run."""

    def __init__(self, kb, root: Path, monkeypatch) -> None:
        self.kb, self.root, self.mp = kb, root, monkeypatch

    def build(self, body: str = "") -> str:
        with contextlib.closing(self.kb.connect(board="web")) as conn:
            return self.kb.create_task(conn, title=TITLE, body=body, assignee=BUILDER)

    def designer(self, *builds, chosen=True, data=PAGE, record=True, **fields) -> int:
        """A designer card above ``builds``. With ``chosen`` the owner picks the first of its two options; then it
        stores ``data`` as full-page.html and completes, its metadata holding the design_pin record (``fields``
        replace the record's own values) unless ``record`` is false. Returns the full page's attachment id."""
        kb = self.kb
        with contextlib.closing(kb.connect(board="web")) as conn:
            card = kb.create_task(conn, title="Design options", assignee=DESIGNER)
            for build in builds:
                kb.link_tasks(conn, card, build)
            run = kb.claim_task(conn, card).current_run_id
            if chosen:
                options = [{"n": n, "sentence": f"Option {n} keeps the page plain.", "pictures": [{
                    "attachment_id": str(kb.store_attachment_bytes(conn, card, f"option-{n}.png", picture(),
                                                                   content_type="image/png", uploaded_by="agent",
                                                                   board="web")), "alt": f"Option {n}"}]}
                           for n in (1, 2)]
                declare_design_options(conn, card, "Which privacy page do you want?", options, board="web",
                                       expected_run_id=run)
                assert accept_choice(conn, card, 1, actor="owner") == "released"
                kb.claim_task(conn, card)
            page = kb.store_attachment_bytes(conn, card, "full-page.html", data, uploaded_by="agent", board="web")
            pin = {"attachment_id": page, "sha256": sha(data), "file": "full-page.html", "pages": ["/privacy/"]}
            assert kb.complete_task(conn, card, summary="full page",
                                    metadata={"design_pin": {**pin, **fields}} if record else None)
        return page

    def review(self, build: str) -> None:
        """The builder's run, its review request, the reviewer's claim, and the environment the dispatcher gives
        that review run: the board's own database and workspaces pinned, a profile home under the root."""
        kb, board = self.kb, self.root / "kanban" / "boards" / "web"
        for var in ENV:
            self.mp.delenv(var, raising=False)
        with contextlib.closing(kb.connect(board="web")) as conn:
            first = kb.claim_task(conn, build)
            assert kb.request_review(conn, build, summary="ready", reviewer=REVIEWER,
                                     expected_run_id=first.current_run_id)
            task = kb.claim_review_task(conn, build, claimer=f"{REVIEWER}:1")
        for key, value in {
            "HERMES_KANBAN_DB": str(board / "kanban.db"), "HERMES_KANBAN_WORKSPACES_ROOT": str(board / "workspaces"),
            "HERMES_KANBAN_BOARD": "web", "HERMES_KANBAN_TASK": build, "HERMES_PROFILE": REVIEWER,
            "HERMES_KANBAN_RUN_ID": str(task.current_run_id), "HERMES_KANBAN_CLAIM_LOCK": task.claim_lock or "",
        }.items():
            self.mp.setenv(key, value)

    def raw(self, sql: str, *params, board="web"):
        """A separate read-only connection to the board's own file."""
        path = self.root / "kanban" / "boards" / board / "kanban.db"
        with contextlib.closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as conn:
            return conn.execute(sql, params).fetchall()


@pytest.fixture
def kanban(tmp_path, monkeypatch):
    from hermes_cli import kanban_db as kb

    root = tmp_path / "root"
    home = root / "profiles" / REVIEWER
    home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    for var in ("HERMES_KANBAN_HOME", "HERMES_KANBAN_ATTACHMENTS_ROOT", "HERMES_DELEGATED_CHILD_CONTEXT", *ENV):
        monkeypatch.delenv(var, raising=False)
    for board in ("web", "elsewhere"):
        kb.create_board(board)
        kb.connect(board=board).close()
    (home / "config.yaml").write_text(json.dumps({"design_browser": {"boards": ["web", "elsewhere"]}}),
                                      encoding="utf-8")
    return Kanban(kb, root, monkeypatch)


def call(action: str = "prepare") -> dict:
    args = {"action": action, "captures": []} if action == "judge" else {"action": action}
    result = registry.dispatch("design_compare", args, task_id="design-compare-test")
    assert isinstance(result, str), result
    return json.loads(result)


def compared() -> dict:
    """What prepare compares: the prototype and the pages."""
    result = call()
    assert "error" not in result, result
    return {"prototype": result["prototype"], "pages": result["pages"]}


def refused(kanban, answer) -> None:
    """prepare and judge both give ``answer``, and no kit file is stored."""
    for action in ("prepare", "judge"):
        assert call(action) == (answer if "error" in answer else {"action": action, **answer})
    assert kanban.raw("SELECT COUNT(*) FROM task_attachments WHERE uploaded_by = 'design_compare'") == [(0,)]


@pytest.mark.parametrize("body,pages", [("", ["/privacy/"]), ("Compare pages: /", ["/"])],
                         ids=["no-body-lines", "pages-line-in-body"])
def test_a_chosen_parent_record_gives_the_pin_and_the_body_line_still_gives_the_pages(kanban, body, pages):
    build = kanban.build(body)
    page = kanban.designer(build)
    kanban.review(build)
    assert compared() == {"prototype": {"attachment_id": page, "sha256": sha(PAGE)}, "pages": pages}


def test_a_record_whose_hash_differs_from_the_file_stops_with_the_existing_question(kanban):
    build = kanban.build()
    kanban.designer(build, sha256=sha(OTHER))
    kanban.review(build)
    refused(kanban, STOP)


@pytest.mark.parametrize("fields", [{"record": False}, {"sha256": sha(PAGE)[:63]}, {"file": None}],
                         ids=["no-record", "short-hash", "no-file"])
def test_a_chosen_design_without_a_readable_full_page_record_is_refused(kanban, fields):
    build = kanban.build()
    kanban.designer(build, **fields)
    kanban.review(build)
    refused(kanban, {"error": NO_RECORD})


def test_two_chosen_parents_are_refused(kanban):
    build = kanban.build()
    kanban.designer(build)
    kanban.designer(build, data=OTHER)
    kanban.review(build)
    refused(kanban, {"error": NO_RECORD})


def test_a_parent_without_the_owners_choice_is_never_read(kanban):
    alone, beside = kanban.build(), kanban.build()
    kanban.designer(alone, beside, chosen=False, data=OTHER)
    page = kanban.designer(beside)
    kanban.review(alone)
    refused(kanban, {"error": NO_PIN})
    kanban.review(beside)
    assert compared()["prototype"] == {"attachment_id": page, "sha256": sha(PAGE)}


def test_a_pin_line_in_the_body_wins_over_the_record(kanban):
    pinned, unpinned = kanban.build(), kanban.build()
    other = kanban.designer(pinned, unpinned, chosen=False, data=OTHER)
    page = kanban.designer(pinned, unpinned)
    with contextlib.closing(kanban.kb.connect(board="web")) as conn:
        conn.execute("UPDATE tasks SET body = ? WHERE id = ?", (
            f"Prototype pin: attachment={other} sha256={sha(OTHER)} file=full-page.html\nCompare pages: /", pinned))
        conn.commit()
    kanban.review(pinned)
    assert compared() == {"prototype": {"attachment_id": other, "sha256": sha(OTHER)}, "pages": ["/"]}
    kanban.review(unpinned)
    assert compared() == {"prototype": {"attachment_id": page, "sha256": sha(PAGE)}, "pages": ["/privacy/"]}


@pytest.mark.parametrize("owner", ["sibling-parent", "build-card"])
def test_a_record_naming_a_file_that_is_not_on_its_own_designer_card_stops(kanban, owner):
    build = kanban.build()
    if owner == "sibling-parent":
        page = kanban.designer(build, chosen=False)
    else:
        with contextlib.closing(kanban.kb.connect(board="web")) as conn:
            page = kanban.kb.store_attachment_bytes(conn, build, "full-page.html", PAGE, uploaded_by="agent",
                                                    board="web")
    kanban.designer(build, attachment_id=page)
    kanban.review(build)
    refused(kanban, STOP)


# ── The dispatcher's worker environment ──────────────────────────────────────


def test_under_the_dispatcher_environment_the_record_is_read_on_the_workers_board_under_the_root(kanban):
    build = kanban.build()
    kanban.designer(build, chosen=False, data=OTHER)
    kanban.designer(build)
    kanban.review(build)
    home, boards = Path(os.environ["HERMES_HOME"]), kanban.root / "kanban" / "boards"
    assert home.parent == kanban.root / "profiles"
    assert os.environ["HERMES_KANBAN_DB"] == str(boards / "web" / "kanban.db")
    result = call()
    assert "error" not in result, result

    # The independent count: the one chosen parent and its latest completed record, read from the board's file.
    (card,), = kanban.raw("SELECT DISTINCT l.parent_id FROM task_links l JOIN task_events e ON e.task_id = "
                          "l.parent_id WHERE l.child_id = ? AND e.kind = 'design_option_chosen'", build)
    (record,), = kanban.raw("SELECT json_extract(metadata, '$.design_pin') FROM task_runs WHERE task_id = ? AND "
                            "outcome = 'completed' ORDER BY started_at DESC, id DESC LIMIT 1", card)
    record = json.loads(record)
    (owner, path), = kanban.raw("SELECT task_id, stored_path FROM task_attachments WHERE id = ?",
                                record["attachment_id"])
    assert owner == card and Path(path).parent == boards / "web" / "attachments" / card
    assert result["prototype"] == {"attachment_id": record["attachment_id"], "sha256": sha(Path(path).read_bytes())}
    assert result["pages"] == record["pages"]
    kit = kanban.raw("SELECT stored_path FROM task_attachments WHERE task_id = ? AND uploaded_by = 'design_compare'",
                     build)
    assert len(kit) == len(result["kit"]) == 2
    assert {Path(p).parent for (p,) in kit} == {boards / "web" / "attachments" / build}
    assert not (home / "kanban").exists() and not (home / "kanban.db").exists()
    assert kanban.raw("SELECT COUNT(*) FROM task_attachments", board="elsewhere") == [(0,)]
    assert kanban.kb.kanban_home() == kanban.root
    assert {"web", "elsewhere"} <= {b["slug"] for b in kanban.kb.list_boards()}
    assert {kanban.kb.board_dir(slug) for slug in ("web", "elsewhere")} == {boards / "web", boards / "elsewhere"}
