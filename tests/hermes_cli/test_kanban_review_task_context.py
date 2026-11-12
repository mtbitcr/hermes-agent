"""A task the daily review created reads only its own card.

The daily review makes its project-review and portfolio-review tasks with
``created_by == "project-review"``. The context ``build_worker_context`` builds
for such a task must hold no title, summary or other text of any other task: not
the ``## Recent work by @<assignee>`` block, which lists the assignee's completed
runs on other tasks (another Project's review among them), and not the
``## Parent task results`` block. An ordinary task keeps both blocks.

The rows come from the real kernel calls (create, claim, complete, link). The
context is read three ways: the function itself, ``hermes kanban context``, and
the worker's own ``kanban_show`` call under the exact environment
``kanban_db._default_spawn`` injects (only ``Popen`` captured). The other
Project's text is also counted straight from the board's sqlite file, so its
absence from a context means something.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import subprocess
from contextlib import closing
from pathlib import Path

import pytest

BOARD = "daily-project-review"
ASSIGNEE = "raphael"
PROFILE_CONFIG = "platform_toolsets:\n  cli:\n    - file\n    - terminal\n    - web\n"
PEER_KEY = "project-review:prj_beta:2026-11-09:0"
# Each place a leak could come from holds its own text: the peer's title, its run summary, its run metadata.
PEER_TITLE = "Daily review of project beta-orchard"
PEER_SUMMARY = "beta-orchard budget plan awaits the owner"
PEER_HEADLINE = "beta-orchard hidden headline"
PEER_METADATA = {"project_review": {"brief": {"headline": PEER_HEADLINE}}}
OWN_BODY = "Review project alpha-grove from the frozen input."
OWN_NOTE = "alpha-grove: the owner wants the budget angle first"
REVIEWS = pytest.mark.parametrize(
    ("key", "title"),
    [
        ("project-review:prj_alpha:2026-11-09:0", "Daily review of project alpha-grove"),
        ("project-review:portfolio:2026-11-09:0", "Daily portfolio review across alpha-grove"),
    ],
    ids=["project-review", "portfolio-review"],
)


@pytest.fixture
def root(tmp_path, monkeypatch):
    """A fresh shared root that holds the board, with the assignee's profile under it."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    for key in [k for k in os.environ if k.startswith("HERMES_KANBAN_")]:
        monkeypatch.delenv(key)
    monkeypatch.delenv("HERMES_PROFILE", raising=False)
    monkeypatch.delenv("HERMES_SESSION_ID", raising=False)

    from hermes_cli import kanban_db as kb

    kb._INITIALIZED_PATHS.clear()
    kb.create_board(BOARD)
    profile = home / "profiles" / ASSIGNEE
    profile.mkdir(parents=True)
    (profile / "config.yaml").write_text(PROFILE_CONFIG, encoding="utf-8")
    return home


@pytest.fixture
def kb(root):
    from hermes_cli import kanban_db

    return kanban_db


def _finished(kb, conn, title, summary, *, created_by, assignee=ASSIGNEE, key=None, metadata=None) -> str:
    """A task the kernel created, claimed and completed, so its run is in the history."""
    tid = kb.create_task(
        conn, title=title, body=f"input of {title}", assignee=assignee,
        created_by=created_by, idempotency_key=key, board=BOARD,
    )
    assert kb.claim_task(conn, tid) is not None
    assert kb.complete_task(conn, tid, summary=summary, metadata=metadata)
    return tid


def _peer_review(kb, conn) -> str:
    """The completed review of another Project, by the same assignee."""
    return _finished(
        kb, conn, PEER_TITLE, PEER_SUMMARY,
        created_by=kb.REVIEW_TASK_CREATOR, key=PEER_KEY, metadata=PEER_METADATA,
    )


def _review_task(kb, conn, key, title) -> str:
    """The review task, ready to be claimed, with its own body, attachment and comment."""
    tid = kb.create_task(
        conn, title=title, body=OWN_BODY, assignee=ASSIGNEE,
        created_by=kb.REVIEW_TASK_CREATOR, idempotency_key=key, board=BOARD,
    )
    kb.store_attachment_bytes(
        conn, tid, "brief-input.md", b"frozen review input\n",
        content_type="text/plain", uploaded_by="owner", board=BOARD,
    )
    kb.add_comment(conn, tid, "owner", OWN_NOTE)
    return tid


def _readonly(db: Path, sql: str, *params):
    with closing(sqlite3.connect(f"{db.resolve().as_uri()}?mode=ro", uri=True)) as conn:
        return conn.execute(sql, params).fetchall()


def _assert_peer_is_in_the_store(kb, peer: str) -> None:
    """Count the peer's row and its completed run straight from the sqlite file: what a leak would carry is there."""
    db = kb.board_dir(BOARD) / "kanban.db"
    assert _readonly(db, "SELECT COUNT(*) FROM tasks WHERE id = ? AND title = ?", peer, PEER_TITLE) == [(1,)]
    assert _readonly(
        db,
        "SELECT COUNT(*) FROM task_runs WHERE task_id = ? AND profile = ? AND outcome = 'completed' "
        "AND summary = ? AND metadata LIKE ?",
        peer, ASSIGNEE, PEER_SUMMARY, f"%{PEER_HEADLINE}%",
    ) == [(1,)]


def _peer_text(peer: str) -> tuple:
    return (peer, PEER_TITLE, PEER_SUMMARY, PEER_HEADLINE, "beta-orchard")


def _assert_only_own_card(text: str, own: str, title: str, peer: str) -> None:
    for hidden in _peer_text(peer):
        assert hidden not in text, text
    # The rule must not pass by emptying the context: every part of the card's own is still there.
    assert text.startswith(f"# Kanban task {own}: {title}\n"), text
    assert f"## Body\n{OWN_BODY}\n" in text, text
    assert "## Attachments" in text and "brief-input.md" in text, text
    assert "## Comment thread" in text and OWN_NOTE in text, text


def _spawn(kb, task, workspace: Path, monkeypatch):
    """Run the real dispatcher spawn with only Popen captured; return (argv, env)."""
    captured = {}

    class _Proc:
        pid = 4242

    def _popen(cmd, *args, **kwargs):
        kwargs["stdout"].close()  # the worker log the dispatcher opened for the child
        captured["cmd"] = list(cmd)
        captured["env"] = dict(kwargs["env"])
        return _Proc()

    monkeypatch.setattr(kb, "_resolve_hermes_argv", lambda: ["hermes"])
    monkeypatch.setattr(subprocess, "Popen", _popen)
    workspace.mkdir(exist_ok=True)
    assert kb._default_spawn(task, str(workspace), board=BOARD) == 4242
    return captured["cmd"], captured["env"]


def _enter_worker_env(monkeypatch, env: dict) -> None:
    """Make this process the worker: exactly what the dispatcher injected or removed."""
    for key in set(os.environ) - set(env):
        monkeypatch.delenv(key)
    for key, value in env.items():
        if os.environ.get(key) != value:
            monkeypatch.setenv(key, value)


def _call(name: str, args: dict) -> dict:
    import model_tools
    import tools.kanban_tools  # noqa: F401 - registers the kanban tools

    return json.loads(model_tools.handle_function_call(name, args))


# ---------------------------------------------------------------------------
# A review task's context holds its own card only
# ---------------------------------------------------------------------------


@REVIEWS
def test_built_context_of_a_review_task_holds_only_its_own_card(kb, key, title):
    with kb.connect_closing(board=BOARD) as conn:
        peer = _peer_review(kb, conn)
        own = _review_task(kb, conn, key, title)
        built = kb.build_worker_context(conn, own)
    _assert_peer_is_in_the_store(kb, peer)
    _assert_only_own_card(built, own, title, peer)


@REVIEWS
def test_cli_context_of_a_review_task_holds_only_its_own_card(kb, key, title):
    from hermes_cli.kanban import run_slash

    with kb.connect_closing(board=BOARD) as conn:
        peer = _peer_review(kb, conn)
        own = _review_task(kb, conn, key, title)
    _assert_peer_is_in_the_store(kb, peer)
    _assert_only_own_card(run_slash(f"--board {BOARD} context {own}"), own, title, peer)


@REVIEWS
def test_start_text_of_a_review_worker_holds_only_its_own_card(kb, root, tmp_path, monkeypatch, key, title):
    with kb.connect_closing(board=BOARD) as conn:
        peer = _peer_review(kb, conn)
        own = _review_task(kb, conn, key, title)
        task = kb.claim_task(conn, own)
    assert task is not None and task.status == "running"
    _assert_peer_is_in_the_store(kb, peer)

    cmd, env = _spawn(kb, task, tmp_path / "workspace", monkeypatch)
    assert env["HERMES_KANBAN_TASK"] == own
    assert env["HERMES_KANBAN_BOARD"] == BOARD
    assert Path(env["HERMES_KANBAN_DB"]).resolve() == (kb.board_dir(BOARD) / "kanban.db").resolve()
    assert Path(env["HERMES_HOME"]).resolve() == (root / "profiles" / ASSIGNEE).resolve()
    # What the dispatcher hands the process at start (prompt and argv, then env) holds none of the peer's text.
    assert f"work kanban task {own}" in cmd, cmd
    for text in (*cmd, *env.values()):
        for hidden in _peer_text(peer):
            assert hidden not in text, (hidden, text)

    # The text the worker reads at start is the reply of its own kanban_show call.
    _enter_worker_env(monkeypatch, env)
    shown = _call("kanban_show", {})
    assert shown["task"]["id"] == own, shown
    _assert_only_own_card(shown["worker_context"], own, title, peer)
    for hidden in _peer_text(peer):
        assert hidden not in json.dumps(shown), shown


def test_finished_parents_reach_an_ordinary_child_but_not_a_review_task(kb):
    # One more parent than the section shows in full: its overflow line names the parent's title.
    count = kb._CTX_MAX_PARENT_RESULTS + 1
    with kb.connect_closing(board=BOARD) as conn:
        parents = [
            _finished(
                kb, conn, f"Parent {i:02d} of gamma-meadow", f"gamma-meadow result {i:02d}",
                created_by="owner", assignee="mira", metadata={"handoff": f"gamma-meadow detail {i:02d}"},
            )
            for i in range(count)
        ]
        review = _review_task(kb, conn, "project-review:prj_alpha:2026-11-09:0", "Daily review of project alpha-grove")
        ordinary = kb.create_task(
            conn, title="Follow-up of the parents", body=OWN_BODY, assignee=ASSIGNEE,
            created_by="owner", board=BOARD,
        )
        for parent in parents:
            kb.link_tasks(conn, parent, review)
            kb.link_tasks(conn, parent, ordinary)
        review_text = kb.build_worker_context(conn, review)
        ordinary_text = kb.build_worker_context(conn, ordinary)

    # The links and the finished parents are really in the sqlite file, for both children.
    db = kb.board_dir(BOARD) / "kanban.db"
    for child in (review, ordinary):
        assert _readonly(db, "SELECT COUNT(*) FROM task_links WHERE child_id = ?", child) == [(count,)]
    assert _readonly(db, "SELECT COUNT(*) FROM tasks WHERE status = 'done' AND title LIKE '%gamma-meadow'") == [(count,)]

    # The same links on an ordinary child still show the results, and name the overflow parent's title.
    assert "## Parent task results" in ordinary_text, ordinary_text
    assert all(parent in ordinary_text for parent in parents), ordinary_text
    assert sum(f"gamma-meadow result {i:02d}" in ordinary_text for i in range(count)) == kb._CTX_MAX_PARENT_RESULTS
    assert sum(f"gamma-meadow detail {i:02d}" in ordinary_text for i in range(count)) == kb._CTX_MAX_PARENT_RESULTS
    assert sum(f"Parent {i:02d} of gamma-meadow" in ordinary_text for i in range(count)) == 1

    assert "## Parent task results" not in review_text, review_text
    assert "gamma-meadow" not in review_text, review_text
    assert not any(parent in review_text for parent in parents), review_text
    assert review_text.startswith("# Kanban task ") and f"## Body\n{OWN_BODY}\n" in review_text


def test_ordinary_task_keeps_its_recent_work_block(kb):
    with kb.connect_closing(board=BOARD) as conn:
        ordinary_peer = _finished(
            kb, conn, "Fix the login form", "login form fixed\nsecond line of the summary", created_by="owner",
        )
        review_peer = _peer_review(kb, conn)
        # Titled like a review, by the same assignee, but made by someone else: only the creator marks a review task.
        own = kb.create_task(
            conn, title="Daily review of project alpha-grove", body=OWN_BODY, assignee=ASSIGNEE,
            created_by="owner", board=BOARD,
        )
        text = kb.build_worker_context(conn, own)

    assert f"\n## Recent work by @{ASSIGNEE}\n" in text, text
    for tid, title, first in (
        (ordinary_peer, "Fix the login form", "login form fixed"),
        (review_peer, PEER_TITLE, PEER_SUMMARY),
    ):
        line = "^" + re.escape(f"- {tid} — {title} (") + r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}, [^)]+" + re.escape(f"): {first}") + "$"
        assert re.search(line, text, re.MULTILINE), text
