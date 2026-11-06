"""``kanban_offer_design_options`` — real registry dispatch, real boards.

Card 1 of the design options plan, platform test 3 of section G. Every test
runs under the exact environment the dispatcher injects into a worker:
``HERMES_KANBAN_DB`` and ``HERMES_KANBAN_BOARD`` pinned to the worker's own
board, ``HERMES_KANBAN_TASK`` set to its card and ``HERMES_HOME`` pointing at
its profile home under the root, with two more boards beside its own. The
designer attaches its pictures to its own card with the real
``kanban_attach``, then offers them through the registry. The tool must
resolve the card's own board and that card's attachments, and refuse any card
other than the worker's own current card. Each proof is read again from the
board store files by a separate read-only connection.
"""

from __future__ import annotations

import base64
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

TOOL = "kanban_offer_design_options"
DECLARED_EVENT = "design_options_declared"
QUESTION = "Which home page should Raphael build?"
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 40
JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 60
BOARDS = ("default", "own-board", "other-board")


# ---------------------------------------------------------------------------
# Copied verbatim from tests/tools/test_kanban_tools.py (copied, not imported,
# so this file stands on its own).
# ---------------------------------------------------------------------------

def _kanban_worker_schema_names(monkeypatch):
    """Tool names a dispatcher-spawned worker actually sees.

    Mirrors model_tools' worker path: the ``kanban`` toolset is folded in for
    a dispatcher-owned worker, then each entry's own check_fn decides.
    """
    import tools.kanban_tools  # noqa: F401  (ensure registered)
    from tools.registry import invalidate_check_fn_cache, registry
    from toolsets import resolve_toolset

    invalidate_check_fn_cache()
    schema = registry.get_definitions(set(resolve_toolset("kanban")), quiet=True)
    names = {s["function"].get("name") for s in schema if "function" in s}
    invalidate_check_fn_cache()
    return {n for n in names if n and n.startswith("kanban_")}


# ---------------------------------------------------------------------------
# Helpers for this file
# ---------------------------------------------------------------------------

def _store_path(slug):
    """The board's own store file, resolved from its slug alone."""
    from hermes_cli import kanban_db as kb

    if slug == kb.DEFAULT_BOARD:
        return kb.kanban_home() / "kanban.db"
    return kb.board_dir(slug) / "kanban.db"


def _claimed_card(slug, title):
    """One card on board ``slug``, claimed by the dispatcher and running."""
    from hermes_cli import kanban_db as kb

    conn = kb.connect(board=slug)
    try:
        task_id = kb.create_task(conn, title=title, assignee="designer", board=slug)
        claimed = kb.claim_task(conn, task_id)
    finally:
        conn.close()
    assert claimed is not None and claimed.status == "running"
    return claimed


def _read(store, query, *params):
    """Rows of ``query`` as a separate read-only connection to ``store`` sees them."""
    conn = sqlite3.connect(f"{Path(store).resolve().as_uri()}?mode=ro", uri=True)
    try:
        return conn.execute(query, params).fetchall()
    finally:
        conn.close()


def _board_state(stores):
    """Per board: each card's (id, status, block_kind) and its declaration events."""
    return {
        slug: (
            _read(store, "SELECT id, status, block_kind FROM tasks ORDER BY id"),
            _read(
                store, "SELECT task_id, payload FROM task_events WHERE kind = ? ORDER BY id",
                DECLARED_EVENT,
            ),
        )
        for slug, store in stores.items()
    }


def _dispatch(name, args):
    from tools.registry import registry

    return json.loads(registry.dispatch(name, args))


def _attach(filename, data, content_type):
    """The designer attaches one picture to its own card with ``kanban_attach``."""
    out = _dispatch("kanban_attach", {
        "filename": filename,
        "content_base64": base64.b64encode(data).decode("ascii"),
        "content_type": content_type,
    })
    assert out.get("ok") is True, out
    return out["attachment_id"]


def _options(*attachment_ids):
    sentences = (
        "A calm layout with a large photo at the top.",
        "A bold layout with a big title and two buttons.",
    )
    alts = (
        "The home page with a wide photo above a short welcome.",
        "The home page with a large title and two buttons below it.",
    )
    return [
        {
            "n": n,
            "sentence": sentences[n - 1],
            "pictures": [{"attachment_id": str(attachment_id), "alt": alts[n - 1]}],
        }
        for n, attachment_id in enumerate(attachment_ids, start=1)
    ]


@pytest.fixture
def worker(tmp_path, monkeypatch):
    """Three boards under one root and the env the dispatcher's spawn gives
    the designer for its card on ``own-board``.
    """
    from hermes_cli import kanban_db as kb
    from hermes_cli.profiles import get_active_profile_name, resolve_profile_env
    from tools.registry import discover_builtin_tools

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    root = tmp_path / ".hermes"
    root.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(root))
    for key in (
        "HERMES_KANBAN_TASK", "HERMES_KANBAN_DB", "HERMES_KANBAN_BOARD",
        "HERMES_KANBAN_RUN_ID", "HERMES_KANBAN_CLAIM_LOCK",
        "HERMES_KANBAN_WORKSPACE", "HERMES_KANBAN_WORKSPACES_ROOT",
        "HERMES_KANBAN_ATTACHMENTS_ROOT", "HERMES_PROFILE", "HERMES_SESSION_ID",
    ):
        monkeypatch.delenv(key, raising=False)

    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    kb.create_board("own-board")
    kb.create_board("other-board")
    mine = _claimed_card("own-board", "home page design")
    sibling = _claimed_card("own-board", "menu page design")
    foreign = _claimed_card("other-board", "shop page design")
    default_card = _claimed_card(kb.DEFAULT_BOARD, "about page design")
    stores = {slug: _store_path(slug) for slug in BOARDS}
    workspaces = kb.workspaces_root(board="own-board")

    (root / "profiles" / "designer").mkdir(parents=True)
    profile_home = Path(resolve_profile_env("designer"))
    # The env the dispatcher's spawn gives a worker for a card on own-board.
    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    monkeypatch.setenv("HERMES_KANBAN_TASK", mine.id)
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACE", str(workspaces / mine.id))
    monkeypatch.setenv("HERMES_SESSION_SOURCE", "kanban")
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(mine.current_run_id))
    monkeypatch.setenv("HERMES_KANBAN_CLAIM_LOCK", mine.claim_lock)
    monkeypatch.setenv("HERMES_KANBAN_DB", str(stores["own-board"]))
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACES_ROOT", str(workspaces))
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "own-board")
    monkeypatch.setenv("HERMES_PROFILE", "designer")

    # Positive controls: the pin is live (the resolver hands back the own
    # store for any slug), the profile resolves from the env alone, the
    # built-in discovery imports the tool's module by name, and the worker's
    # schema lists the tool.
    assert kb.kanban_db_path(board="other-board") == stores["own-board"]
    assert get_active_profile_name() == "designer"
    assert "tools.kanban_design_options" in discover_builtin_tools()
    assert TOOL in _kanban_worker_schema_names(monkeypatch)
    return SimpleNamespace(
        root=root, profile_home=profile_home, stores=stores, mine=mine,
        sibling=sibling, foreign=foreign, default_card=default_card,
    )


def test_the_worker_offers_options_on_its_own_card_on_its_own_board(worker):
    first = _attach("calm.png", PNG, "image/png")
    second = _attach("bold.jpg", JPEG, "image/jpeg")
    before = _board_state(worker.stores)

    out = _dispatch(TOOL, {"question": QUESTION, "options": _options(first, second)})

    assert out == {
        "ok": True,
        "task_id": worker.mine.id,
        "status": "blocked",
        "block_kind": "needs_input",
        "declared": True,
    }
    after = _board_state(worker.stores)
    # Only the worker's own board changed: its card alone is held, with one
    # declaration; every other card on every board is as it was.
    assert {slug: after[slug] for slug in ("default", "other-board")} == {
        slug: before[slug] for slug in ("default", "other-board")
    }
    own_cards, own_declarations = after["own-board"]
    assert own_cards == sorted([
        (worker.mine.id, "blocked", "needs_input"),
        (worker.sibling.id, "running", None),
    ])
    ((declared_on, payload),) = own_declarations
    assert declared_on == worker.mine.id
    pictures = [option["pictures"][0] for option in json.loads(payload)["options"]]
    # The pictures resolved are the card's own attachments, as its board
    # store records them and as their files under the root hold them.
    stored = _read(
        worker.stores["own-board"],
        "SELECT id, content_type, size, stored_path FROM task_attachments "
        "WHERE task_id = ? ORDER BY id",
        worker.mine.id,
    )
    assert [
        {"attachment_id": str(row[0]), "media_type": row[1], "bytes": row[2]}
        for row in stored
    ] == [
        {key: picture[key] for key in ("attachment_id", "media_type", "bytes")}
        for picture in pictures
    ]
    own_attachments = (worker.root / "kanban" / "boards" / "own-board" / "attachments").resolve()
    assert [Path(row[3]).read_bytes() for row in stored] == [PNG, JPEG]
    assert all(Path(row[3]).resolve().parent == own_attachments / worker.mine.id for row in stored)
    assert not (worker.profile_home / "kanban").exists()


def test_a_replay_of_the_same_offer_changes_nothing(worker):
    options = _options(_attach("calm.png", PNG, "image/png"), _attach("bold.jpg", JPEG, "image/jpeg"))
    assert _dispatch(TOOL, {"question": QUESTION, "options": options})["declared"] is True
    held = _board_state(worker.stores)

    out = _dispatch(TOOL, {"question": QUESTION, "options": options})

    assert out["ok"] is True and out["declared"] is False
    assert _board_state(worker.stores) == held


@pytest.mark.parametrize("target", ["sibling", "foreign", "default_card"])
def test_the_tool_refuses_any_card_other_than_the_workers_own(worker, target):
    options = _options(_attach("calm.png", PNG, "image/png"), _attach("bold.jpg", JPEG, "image/jpeg"))
    before = _board_state(worker.stores)

    out = _dispatch(TOOL, {
        "task_id": getattr(worker, target).id, "question": QUESTION, "options": options,
    })

    assert "error" in out and "ok" not in out
    assert _board_state(worker.stores) == before


def test_the_tool_refuses_a_caller_that_is_not_the_cards_worker(worker, monkeypatch):
    options = _options(_attach("calm.png", PNG, "image/png"), _attach("bold.jpg", JPEG, "image/jpeg"))
    before = _board_state(worker.stores)
    # The same process without the dispatcher's card: an orchestrator or a
    # chat session naming the card is not the card's worker.
    monkeypatch.delenv("HERMES_KANBAN_TASK")

    out = _dispatch(TOOL, {
        "task_id": worker.mine.id, "question": QUESTION, "options": options,
    })

    assert "error" in out and "ok" not in out
    assert _board_state(worker.stores) == before


def test_a_refused_offer_names_the_limit_and_writes_nothing(worker):
    first = _attach("calm.png", PNG, "image/png")
    before = _board_state(worker.stores)

    out = _dispatch(TOOL, {"question": QUESTION, "options": _options(first)})

    assert "error" in out and "2 or 3" in out["error"]
    assert _board_state(worker.stores) == before
