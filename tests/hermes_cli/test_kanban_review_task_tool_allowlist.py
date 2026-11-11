"""Operator rule S1: review and portfolio tasks may call only kanban_show and kanban_complete.

The daily project review creates its review and portfolio cards with
``created_by == "project-review"``. The lock applies in code and needs no
flag. The dispatcher spawns such a worker with the ``kanban`` toolset and the
tool limit ``HERMES_KANBAN_ALLOWED_TOOLS``. From that limit ``model_tools``
shows the model exactly the two tools, assembles no Tool Search bridge, and
refuses every other call. Behind it, every other kanban tool's handler still
refuses on its own.

These tests use the real spawn (``kanban_db._default_spawn``, with only
``Popen`` captured), the real CLI parser, the real toolset resolution in
``model_tools``, and real tool calls through
``model_tools.handle_function_call``. The calls run on a board the kernel made,
under the exact environment the dispatcher injects into the worker. The CLI
regression spawns the real ``hermes`` CLI against a loopback model.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from contextlib import closing
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import yaml

BOARD = "daily-project-review"
ASSIGNEE = "raphael"
# A design role, so kanban_view_image's own access gate lets the call reach kanban_read.
DESIGNER = "raphael-designer"
REVIEW_KEY = "project-review:prj_alpha:2026-11-09:0"
PORTFOLIO_KEY = "project-review:portfolio:2026-11-09:0"
REVIEW_KEYS = pytest.mark.parametrize("key", [REVIEW_KEY, PORTFOLIO_KEY], ids=["review", "portfolio"])
PAIR = {"kanban_show", "kanban_complete"}
LIMIT_ENV = "HERMES_KANBAN_ALLOWED_TOOLS"
# The assignee's own CLI tools are wide, so a pin that fell back to them would show.
PROFILE_CONFIG = "platform_toolsets:\n  cli:\n    - file\n    - terminal\n    - web\n"
# Tool Search forced on, with the pair named for deferral: without the limit the bridge shows.
TOOL_SEARCH_ON = "tools:\n  tool_search:\n    enabled: 'on'\n    defer: [kanban_show, kanban_complete]\n"
# The review's discovery probes: each read kanban_read's schema before this limit.
BRIDGE_PROBES = {
    "tool_search": {"queries": ["kanban"]},
    "tool_describe": {"names": ["kanban_read"]},
    "tool_call": {"calls": [{"name": "kanban_read", "arguments": {}}]},
}
REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def root(tmp_path, monkeypatch):
    """A fresh shared root that holds the board, with the profiles under it."""
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
    return home


def _profile(root: Path, name: str, config: str | None = PROFILE_CONFIG) -> Path:
    home = root / "profiles" / name
    home.mkdir(parents=True, exist_ok=True)
    if config is not None:
        (home / "config.yaml").write_text(config, encoding="utf-8")
    return home


def _claimed_task(kb, *, created_by, key=None, assignee=ASSIGNEE):
    """A kernel-made task on the board, claimed, so it has a running run."""
    with kb.connect_closing(board=BOARD) as conn:
        tid = kb.create_task(
            conn, title="daily project review", body="review the project", assignee=assignee,
            created_by=created_by, idempotency_key=key, board=BOARD,
        )
        kb.store_attachment_bytes(
            conn, tid, "brief-input.md", b"frozen review input\n",
            content_type="text/plain", uploaded_by="owner", board=BOARD,
        )
        task = kb.claim_task(conn, tid)
    assert task is not None and task.status == "running" and task.current_run_id is not None
    return task


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


def _pinned_toolsets(cmd: list) -> str:
    assert cmd.count("--toolsets") == 1, cmd
    return cmd[cmd.index("--toolsets") + 1]


def _enter_worker_env(monkeypatch, env: dict) -> None:
    """Make this process the worker: exactly what the dispatcher injected or removed."""
    for key in set(os.environ) - set(env):
        monkeypatch.delenv(key)
    for key, value in env.items():
        if os.environ.get(key) != value:
            monkeypatch.setenv(key, value)


def _register_kanban_tools() -> None:
    """Import every module that registers a kanban_* tool."""
    import model_tools  # noqa: F401
    import tools.kanban_design_options  # noqa: F401
    import tools.kanban_render_drawing  # noqa: F401
    import tools.kanban_tools  # noqa: F401
    import tools.kanban_view_image_tool  # noqa: F401


def _call(name: str, args: dict) -> dict:
    import model_tools

    return json.loads(model_tools.handle_function_call(name, args))


def _advertised(**kwargs) -> set:
    import model_tools

    return {d["function"]["name"] for d in model_tools.get_tool_definitions(quiet_mode=True, **kwargs)}


def _assert_s1_refusal(out: dict, tool: str) -> None:
    error = out.get("error") or ""
    assert out.get("ok") is not True and error.startswith(f"{tool} refused"), out
    assert "operator rule S1" in error and "kanban_show" in error and "kanban_complete" in error, out


def _board_state(db: Path, attachments: Path):
    """Every row of every table of the board, and every attachment file's bytes."""
    conn = sqlite3.connect(f"{db.resolve().as_uri()}?mode=ro", uri=True)
    try:
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name")]
        rows = {t: sorted(repr(r) for r in conn.execute(f'SELECT * FROM "{t}"')) for t in tables}
    finally:
        conn.close()
    files = {p.relative_to(attachments).as_posix(): p.read_bytes() for p in attachments.rglob("*") if p.is_file()}
    assert rows["tasks"] and files, "the comparison must cover a populated board"
    return rows, files


def _worker_on_review_task(kb, root, tmp_path, monkeypatch, *, key, assignee=ASSIGNEE):
    """Spawn a review task's worker and become it, under the dispatcher's own env."""
    task = _claimed_task(kb, created_by="project-review", key=key, assignee=assignee)
    db, attachments = kb.board_dir(BOARD) / "kanban.db", kb.attachments_root(board=BOARD)
    _cmd, env = _spawn(kb, task, tmp_path / "workspace", monkeypatch)
    assert env["HERMES_KANBAN_TASK"] == task.id
    assert env["HERMES_KANBAN_BOARD"] == BOARD
    assert Path(env["HERMES_KANBAN_DB"]).resolve() == db.resolve()
    assert Path(env["HERMES_HOME"]).resolve() == (root / "profiles" / assignee).resolve()
    _enter_worker_env(monkeypatch, env)
    return task, db, attachments


# ---------------------------------------------------------------------------
# The spawn pin and the tool limit
# ---------------------------------------------------------------------------


@REVIEW_KEYS
def test_review_task_worker_is_pinned_to_the_kanban_toolset(root, tmp_path, monkeypatch, key):
    from hermes_cli import kanban_db as kb
    from hermes_cli._parser import build_top_level_parser
    from tools.registry import invalidate_check_fn_cache

    profile = _profile(root, ASSIGNEE, PROFILE_CONFIG + TOOL_SEARCH_ON)
    assert {"file", "terminal", "web"} <= set(kb._resolve_worker_cli_toolsets(str(profile)))
    task = _claimed_task(kb, created_by="project-review", key=key)
    assert task.idempotency_key == key

    cmd, env = _spawn(kb, task, tmp_path / "workspace", monkeypatch)
    assert _pinned_toolsets(cmd) == "kanban"
    assert env[LIMIT_ENV] == ",".join(kb.REVIEW_TASK_TOOLS)

    # The real CLI parser reads the pin as the worker will.
    parser, _subparsers, _chat_parser = build_top_level_parser()
    assert cmd[1:3] == ["-p", ASSIGNEE]
    args = parser.parse_args(cmd[3:])
    assert args.command == "chat"
    toolsets = [name.strip() for name in args.toolsets.split(",")]

    # The tool names model_tools resolves from that pin, in the worker's env.
    _enter_worker_env(monkeypatch, env)
    _register_kanban_tools()
    import model_tools

    ran = []
    execute = model_tools._execute_tool

    def recording_execute(name, *args, **kwargs):
        ran.append(name)
        return execute(name, *args, **kwargs)

    monkeypatch.setattr(model_tools, "_execute_tool", recording_execute)

    def called(name: str, args: dict) -> dict:
        return json.loads(model_tools.handle_function_call(name, args, enabled_toolsets=toolsets))

    invalidate_check_fn_cache()
    try:
        catalog = _advertised(enabled_toolsets=toolsets, skip_tool_search_assembly=True)
        shown = _advertised(enabled_toolsets=toolsets)
        # Every bridge call, and every call of another tool, is refused before anything runs.
        for name, probe in BRIDGE_PROBES.items():
            _assert_s1_refusal(called(name, probe), name)
        for name in ("web_search", "terminal", "kanban_read", "kanban_render_drawing"):
            _assert_s1_refusal(called(name, {}), name)
            _assert_s1_refusal(called("tool_call", {"calls": [{"name": name, "arguments": {}}]}), "tool_call")
        assert ran == []
    finally:
        invalidate_check_fn_cache()
    # The exact-pair contract: the model is offered these two tools and nothing else.
    assert catalog == PAIR
    assert shown == PAIR


@pytest.mark.parametrize("created_by", ["default", None])
def test_other_tasks_keep_their_profile_toolsets(root, tmp_path, monkeypatch, created_by):
    from hermes_cli import kanban_db as kb

    profile = _profile(root, ASSIGNEE)
    task = dataclasses.replace(_claimed_task(kb, created_by="default"), created_by=created_by)
    # A dispatcher that itself carries the limit still never hands it to another task.
    monkeypatch.setenv(LIMIT_ENV, ",".join(kb.REVIEW_TASK_TOOLS))
    cmd, env = _spawn(kb, task, tmp_path / "workspace", monkeypatch)

    assert LIMIT_ENV not in env
    assert env["HERMES_HOME"] == str(profile)
    expected = kb._resolve_worker_cli_toolsets(env["HERMES_HOME"])
    assert expected
    assert _pinned_toolsets(cmd) == ",".join(expected)


def test_the_tool_definitions_memo_follows_the_limit(root, tmp_path, monkeypatch):
    """The same arguments never reuse a memoized result across the limit."""
    from hermes_cli import kanban_db as kb
    from tools.registry import invalidate_check_fn_cache

    _profile(root, ASSIGNEE)
    _register_kanban_tools()
    cmd, env = _spawn(kb, _claimed_task(kb, created_by="default"), tmp_path / "workspace", monkeypatch)
    _enter_worker_env(monkeypatch, env)
    toolsets = _pinned_toolsets(cmd).split(",")

    invalidate_check_fn_cache()
    try:
        full = _advertised(enabled_toolsets=toolsets)
        assert PAIR < full and "terminal" in full
        monkeypatch.setenv(LIMIT_ENV, ",".join(kb.REVIEW_TASK_TOOLS))
        assert _advertised(enabled_toolsets=toolsets) == PAIR
        monkeypatch.delenv(LIMIT_ENV)
        assert _advertised(enabled_toolsets=toolsets) == full
    finally:
        invalidate_check_fn_cache()


@pytest.mark.parametrize(
    "value",
    ["", " , ", "kanban_show;kanban_complete", "kanban_show,kanban_complete,terminal,web_search,kanban_read,tool_search"],
    ids=["empty", "blank", "malformed", "widened"],
)
def test_a_present_limit_never_reaches_past_the_pair(root, tmp_path, monkeypatch, value):
    """The worker's own .env loads with override and could rewrite the limit; a
    rewritten, empty or malformed limit fails closed, even with every toolset."""
    from hermes_cli import kanban_db as kb
    from tools.registry import invalidate_check_fn_cache

    _profile(root, ASSIGNEE, PROFILE_CONFIG + TOOL_SEARCH_ON)
    _register_kanban_tools()
    _cmd, env = _spawn(kb, _claimed_task(kb, created_by="project-review", key=REVIEW_KEY), tmp_path / "w", monkeypatch)
    _enter_worker_env(monkeypatch, {**env, LIMIT_ENV: value})

    invalidate_check_fn_cache()
    try:
        assert _advertised(enabled_toolsets=None) <= PAIR
        for name, probe in BRIDGE_PROBES.items():
            _assert_s1_refusal(_call(name, probe), name)
        for name in ("terminal", "web_search", "kanban_read", "kanban_create"):
            _assert_s1_refusal(_call(name, {}), name)
    finally:
        invalidate_check_fn_cache()


# ---------------------------------------------------------------------------
# The refusal, through the real entry point
# ---------------------------------------------------------------------------


@REVIEW_KEYS
def test_review_task_may_call_only_show_and_complete(root, tmp_path, monkeypatch, key):
    """Runs under the exact env the dispatcher injects: HERMES_KANBAN_DB and
    HERMES_KANBAN_BOARD pinned to the worker's own board, HERMES_KANBAN_TASK set,
    and HERMES_HOME a profile home under the root. model_tools refuses first;
    without its limit, each handler still refuses on its own, reading the task
    there. kanban_show still reads tasks there.

    model_tools.handle_function_call records no board liveness, so the
    comparison excludes nothing. kanban_view_image is covered by its own test
    below. It decides access through kanban_read's handler only after its
    design gate.
    """
    from hermes_cli import kanban_db as kb
    from tools.registry import registry

    _profile(root, ASSIGNEE)
    _register_kanban_tools()
    task, db, attachments = _worker_on_review_task(kb, root, tmp_path, monkeypatch, key=key)

    kanban_tools = {name for name in registry.get_all_tool_names() if name.startswith("kanban_")}
    assert PAIR < kanban_tools
    before = _board_state(db, attachments)
    for name in sorted(kanban_tools - PAIR - {"kanban_view_image"}):
        _assert_s1_refusal(_call(name, {}), name)
    with monkeypatch.context() as handlers_only:
        handlers_only.delenv(LIMIT_ENV)
        for name in sorted(kanban_tools - PAIR - {"kanban_view_image"}):
            _assert_s1_refusal(_call(name, {}), name)
    assert _board_state(db, attachments) == before

    shown = _call("kanban_show", {})
    assert shown["task"]["id"] == task.id, shown
    done = _call("kanban_complete", {"summary": "brief posted"})
    assert done.get("ok") is True, done
    with kb.connect_closing(board=BOARD) as conn:
        assert kb.get_task(conn, task.id).status == "done"


@REVIEW_KEYS
def test_view_image_gets_the_refusal_of_the_read_it_delegates_to(root, tmp_path, monkeypatch, key):
    """model_tools refuses kanban_view_image itself. Without that limit,
    kanban_view_image does its own work first: its design gate (the board
    listed, a design role) and the attachment id. Past those, its access decision
    is kanban_read's handler, called as the worker's own kanban_read. So the
    refusal it returns is kanban_read's S1 refusal, with no change."""
    from hermes_cli import kanban_db as kb

    _profile(root, DESIGNER, PROFILE_CONFIG + f"design_browser:\n  boards:\n    - {BOARD}\n")
    _register_kanban_tools()
    _task, db, attachments = _worker_on_review_task(
        kb, root, tmp_path, monkeypatch, key=key, assignee=DESIGNER,
    )

    before = _board_state(db, attachments)
    _assert_s1_refusal(_call("kanban_view_image", {"attachment_id": 1}), "kanban_view_image")
    monkeypatch.delenv(LIMIT_ENV)
    _assert_s1_refusal(_call("kanban_view_image", {"attachment_id": 1}), "kanban_read")
    assert _board_state(db, attachments) == before


def test_ordinary_running_task_keeps_heartbeat_and_comment(root, tmp_path, monkeypatch):
    from hermes_cli import kanban_db as kb

    _profile(root, ASSIGNEE)
    _register_kanban_tools()
    task = _claimed_task(kb, created_by="default")
    _cmd, env = _spawn(kb, task, tmp_path / "workspace", monkeypatch)
    _enter_worker_env(monkeypatch, env)

    beat = _call("kanban_heartbeat", {"note": "still working"})
    assert beat.get("ok") is True, beat
    note = _call("kanban_comment", {"task_id": task.id, "body": "halfway there"})
    assert note.get("ok") is True, note
    with kb.connect_closing(board=BOARD) as conn:
        assert [c.body for c in kb.list_comments(conn, task.id)] == ["halfway there"]


def _readonly_rows(db: Path, sql: str) -> list:
    with closing(sqlite3.connect(f"{db.resolve().as_uri()}?mode=ro", uri=True)) as conn:
        return conn.execute(sql).fetchall()


def test_status_report_from_a_worker_reaches_every_board_and_the_root_registry(root, tmp_path, monkeypatch):
    """Under the exact env the dispatcher injects (HERMES_KANBAN_DB and
    HERMES_KANBAN_BOARD pinned to the worker's board, HERMES_KANBAN_TASK set,
    HERMES_HOME a profile home under the root), an ordinary task's
    kanban_status_report still reaches every other board and the root project
    registry; its answer is read against an independent count of each store.
    A review task with the same profile is refused."""
    from hermes_cli import kanban_db as kb
    from hermes_cli import projects_db
    from tools.kanban_tools import KANBAN_STATUS_REPORT_NO_PROJECT

    _profile(root, ASSIGNEE, PROFILE_CONFIG + f"kanban:\n  status_report_profiles:\n    - {ASSIGNEE}\n")
    _register_kanban_tools()
    kb.init_db()
    kb.create_board("alpha-board")
    for slug, project in ((kb.DEFAULT_BOARD, "Default Project"), ("alpha-board", "Alpha Project"), (BOARD, None)):
        with closing(projects_db.connect()) as pconn:
            project_id = projects_db.create_project(pconn, name=project, allow_duplicate_path=True) if project else None
        with kb.connect_closing(board=slug) as conn:
            tid = kb.create_task(conn, title=f"waiting on {slug}", assignee="worker", board=slug, project_id=project_id)
            kb.claim_task(conn, tid)
            kb.block_task(conn, tid, reason="needs the owner")

    reports = {}
    for created_by, key in (("default", None), ("project-review", REVIEW_KEY)):
        task = _claimed_task(kb, created_by=created_by, key=key)
        with monkeypatch.context() as worker:
            _cmd, env = _spawn(kb, task, tmp_path / f"workspace-{task.id}", worker)
            assert Path(env["HERMES_KANBAN_DB"]).resolve() == (kb.board_dir(BOARD) / "kanban.db").resolve()
            assert Path(env["HERMES_HOME"]).resolve() == (root / "profiles" / ASSIGNEE).resolve()
            _enter_worker_env(worker, env)
            reports[created_by] = _call("kanban_status_report", {"status": "blocked"})

    # The independent count: every board's own store and the root registry, read directly.
    stores = [root / "kanban.db", *sorted((root / "kanban" / "boards").glob("*/kanban.db"))]
    names = dict(_readonly_rows(root / "projects.db", "SELECT id, name FROM projects"))
    expected = sorted(
        (title, names.get(project_id, KANBAN_STATUS_REPORT_NO_PROJECT))
        for store in stores
        for title, project_id in _readonly_rows(store, "SELECT title, project_id FROM tasks WHERE status = 'blocked'")
    )
    assert len(stores) == 3 and len(expected) == 3 and len(names) == 2
    out = reports["default"]
    assert out.get("boards_scanned") == len(stores) and out["excluded"]["boards_unreadable"] == 0, out
    assert out["count"] == len(expected)
    assert sorted((row["title"], row["project"]) for row in out["cards"]) == expected
    _assert_s1_refusal(reports["project-review"], "kanban_status_report")


def test_the_lock_needs_no_flag(root, tmp_path, monkeypatch):
    """No config file anywhere (the release default): every layer still holds."""
    from hermes_cli import kanban_db as kb

    profile = _profile(root, ASSIGNEE, config=None)
    assert not (root / "config.yaml").exists() and not (profile / "config.yaml").exists()
    _register_kanban_tools()
    task = _claimed_task(kb, created_by="project-review", key=REVIEW_KEY)
    cmd, env = _spawn(kb, task, tmp_path / "workspace", monkeypatch)
    assert _pinned_toolsets(cmd) == "kanban"
    assert env[LIMIT_ENV] == ",".join(kb.REVIEW_TASK_TOOLS)

    _enter_worker_env(monkeypatch, env)
    _assert_s1_refusal(_call("tool_search", BRIDGE_PROBES["tool_search"]), "tool_search")
    _assert_s1_refusal(_call("kanban_create", {"title": "a new card", "assignee": ASSIGNEE}), "kanban_create")
    assert _call("kanban_show", {})["task"]["id"] == task.id


def test_dormant_dispatch_copy_is_unchanged():
    repo = Path(__file__).resolve().parents[2]
    recorded = json.loads((repo / "tests" / "fork_freeze" / "dormant_modules.json").read_text(encoding="utf-8"))
    path = "hermes_cli/kanban_db_dispatch.py"
    assert hashlib.sha256((repo / path).read_bytes()).hexdigest() == recorded[path]


# ---------------------------------------------------------------------------
# The real dispatcher and CLI, against a loopback model
# ---------------------------------------------------------------------------

CLI_MODEL = "s1-loopback-model"
# The agent loop answers a tool it never offered with "does not exist" and stops
# after three turns in a row of only such calls, so a valid kanban_show follows
# each refused call and the run still reaches kanban_complete.
REVIEW_PLAN = [
    ("tool_search", BRIDGE_PROBES["tool_search"]), ("kanban_show", {}),
    ("tool_describe", BRIDGE_PROBES["tool_describe"]), ("kanban_show", {}),
    ("tool_call", BRIDGE_PROBES["tool_call"]), ("kanban_show", {}),
    ("kanban_create", {"title": "a new card", "assignee": ASSIGNEE}), ("kanban_show", {}),
    ("kanban_complete", {"summary": "brief posted"}),
]
ORDINARY_PLAN = [
    ("kanban_heartbeat", {"note": "still working"}),
    ("terminal", {"command": "printf S1_TERMINAL_PROBE", "timeout": 10}),
    ("kanban_show", {}),
    ("kanban_complete", {"summary": "ordinary work done"}),
]
real_cli = pytest.mark.skipif(os.name != "posix", reason="waits on the spawned worker's process group")


class _LoopbackModel:
    """An OpenAI-compatible model on 127.0.0.1. It answers each request that
    carries tools with the next scripted call. Auxiliary requests carry no
    tools and do not advance the script."""

    def __init__(self, plan):
        self.plan, self.requests, self._lock = plan, [], threading.Lock()
        model = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, raw: bytes, content_type: str) -> None:
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self):
                body = {"object": "list", "data": [{"id": CLI_MODEL, "context_length": 128000}]}
                self._send(json.dumps(body).encode(), "application/json")

            def do_POST(self):
                self._send(*model.answer(json.loads(self.rfile.read(int(self.headers["Content-Length"])))))

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def answer(self, request: dict):
        step = len(self.plan)
        if request.get("tools"):
            with self._lock:
                step = len(self.requests)
                self.requests.append(request)
        if step < len(self.plan):
            name, args = self.plan[step]
            call = {"id": f"call_{step}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
            message, finish = {"role": "assistant", "content": None, "tool_calls": [call]}, "tool_calls"
        else:
            message, finish = {"role": "assistant", "content": "Finished."}, "stop"
        usage = {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        head = {"id": "s1", "created": 1, "model": CLI_MODEL}
        if not request.get("stream"):
            body = {**head, "object": "chat.completion", "usage": usage,
                    "choices": [{"index": 0, "message": message, "finish_reason": finish}]}
            return json.dumps(body).encode(), "application/json"
        delta = {k: v for k, v in message.items() if k != "role"}
        if "tool_calls" in delta:
            delta["tool_calls"] = [dict(delta["tool_calls"][0], index=0)]
        chunks = [
            {**head, "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": delta, "finish_reason": None}]},
            {**head, "object": "chat.completion.chunk", "usage": usage,
             "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]},
        ]
        return ("".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n").encode(), "text/event-stream"

    def advertised(self) -> list:
        return [{t["function"]["name"] for t in request["tools"]} for request in self.requests]

    def results(self) -> dict:
        return {m.get("tool_call_id"): m.get("content") for request in self.requests
                for m in request.get("messages", []) if m.get("role") == "tool"}

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


def _real_dispatcher_env(monkeypatch, tmp_path) -> None:
    """The dispatcher's own env for a real worker: this tree's CLI on this
    interpreter, no provider credentials, and no network route but loopback."""
    for key in [k for k in os.environ if k.startswith(("OPENAI_", "ANTHROPIC_", "OPENROUTER_", "AUXILIARY_"))]:
        monkeypatch.delenv(key)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_BIN", str(REPO / "hermes"))
    monkeypatch.setenv("PATH", f"{Path(sys.executable).parent}{os.pathsep}{os.environ.get('PATH', '')}")
    monkeypatch.setenv("PYTHONPATH", str(REPO))
    monkeypatch.setenv("PYTHONUNBUFFERED", "1")
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.setenv(key, "http://127.0.0.1:9")
    for key in ("NO_PROXY", "no_proxy"):
        monkeypatch.setenv(key, "127.0.0.1,localhost")


def _run_real_worker(kb, root, tmp_path, task, plan, *, timeout: float = 240):
    """Spawn ``task``'s worker through kanban_db._default_spawn and the real
    ``hermes`` CLI; return (model, exit code or "timeout", worker log)."""
    model = _LoopbackModel(plan)
    try:
        config = {
            "model": {"provider": "custom", "default": CLI_MODEL, "context_length": 128000,
                      "base_url": f"http://127.0.0.1:{model.server.server_port}/v1"},
            "platform_toolsets": {"cli": ["file", "terminal", "web"]},
            "display": {"streaming": False, "interface": "cli"},
            "agent": {"max_turns": 16},
            "memory": {"enabled": False}, "curator": {"enabled": False},
            "compression": {"enabled": False}, "checkpoints": {"enabled": False},
        }
        _profile(root, ASSIGNEE, yaml.safe_dump(config))
        workspace = tmp_path / f"workspace-{task.id}"
        workspace.mkdir()
        pid = kb._default_spawn(task, str(workspace), board=BOARD)
        deadline, code = time.monotonic() + timeout, "timeout"
        while time.monotonic() < deadline:
            done, status = os.waitpid(pid, os.WNOHANG)
            if done:
                code = os.waitstatus_to_exitcode(status)
                break
            time.sleep(0.2)
        else:
            os.killpg(pid, signal.SIGKILL)
            os.waitpid(pid, 0)
    finally:
        model.close()
    log = kb.worker_logs_dir(board=BOARD) / f"{task.id}.log"
    return model, code, log.read_text(encoding="utf-8", errors="replace")[-6000:] if log.exists() else ""


@real_cli
@REVIEW_KEYS
def test_real_cli_review_worker_is_offered_and_runs_only_the_pair(root, tmp_path, monkeypatch, key):
    from hermes_cli import kanban_db as kb

    _real_dispatcher_env(monkeypatch, tmp_path)
    task = _claimed_task(kb, created_by="project-review", key=key)
    with kb.connect_closing(board=BOARD) as conn:
        cards = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]

    model, code, log = _run_real_worker(kb, root, tmp_path, task, REVIEW_PLAN)

    with kb.connect_closing(board=BOARD) as conn:
        final = kb.get_task(conn, task.id)
        assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == cards
    assert code == 0 and final.status == "done", log
    assert len(model.requests) >= len(REVIEW_PLAN), log
    # Every request of the worker's own session offers exactly the pair.
    assert all(names == PAIR for names in model.advertised()), model.advertised()
    results = model.results()
    for step, (name, _args) in enumerate(REVIEW_PLAN):
        result = results[f"call_{step}"]
        if name in PAIR:
            assert "does not exist" not in result and "refused" not in result, (name, result)
        else:
            # Never offered, so the agent loop rejects it: no catalog read, no schema, no card.
            assert result.startswith(f"Tool '{name}' does not exist. Available tools: kanban_complete, kanban_show"), result
            assert "kanban_read" not in result, result
    assert task.id in results["call_1"]


@real_cli
def test_real_cli_ordinary_worker_keeps_every_tool_of_its_toolsets(root, tmp_path, monkeypatch, record_property):
    """The control: an ordinary task's worker is offered every tool its own
    toolsets resolve to, the pair among them, and its calls run."""
    from hermes_cli import kanban_db as kb
    from tools.registry import invalidate_check_fn_cache

    _real_dispatcher_env(monkeypatch, tmp_path)
    _register_kanban_tools()
    task = _claimed_task(kb, created_by="default")

    model, code, log = _run_real_worker(kb, root, tmp_path, task, ORDINARY_PLAN)

    # The same resolution in this process, under the env the dispatcher gives that worker.
    with monkeypatch.context() as worker:
        cmd, env = _spawn(kb, task, tmp_path / "resolution", worker)
        assert LIMIT_ENV not in env
        _enter_worker_env(worker, env)
        invalidate_check_fn_cache()
        try:
            expected = _advertised(enabled_toolsets=_pinned_toolsets(cmd).split(","))
        finally:
            invalidate_check_fn_cache()

    with kb.connect_closing(board=BOARD) as conn:
        assert kb.get_task(conn, task.id).status == "done", log
    assert code == 0, log
    advertised = model.advertised()
    assert len(advertised) >= len(ORDINARY_PLAN), log
    assert all(names == expected for names in advertised), (advertised, expected)
    assert PAIR < expected and {"kanban_heartbeat", "kanban_comment", "terminal"} <= expected
    results = model.results()
    assert json.loads(results["call_0"]).get("ok") is True, results["call_0"]
    assert "S1_TERMINAL_PROBE" in results["call_1"], results["call_1"]
    record_property("advertised_tools", len(advertised[0]))
