"""Board setting and role gate of the look-only design browser (design-loop plan v2, B4).

Every case runs under the environment the dispatcher gives a worker: ``HERMES_KANBAN_DB`` and
``HERMES_KANBAN_BOARD`` pinned to the worker's own board, ``HERMES_KANBAN_TASK``,
``HERMES_KANBAN_RUN_ID`` and ``HERMES_KANBAN_CLAIM_LOCK`` set, and ``HERMES_HOME`` a profile home
under the root. Each case asserts the gate's own answer and the tool's answer through the
registry. The tool's ``design_browser.cdp_url`` is a fake CDP endpoint that counts the
connections it accepts and closes each one at once, so "refused" is also read against that
independent count: nothing was connected to.
"""

from __future__ import annotations

import asyncio
import contextlib
import http.server
import json
import os
import re
import shutil
import socket
import sqlite3
import ssl
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

import pytest

import toolsets
from tools import design_browser_tool, design_url_policy, url_safety
from tools.design_access import (
    BOARD_NOT_LISTED, BUILDER, DESIGNER, NO_BOARDS, NO_DESIGN_ROLE, NO_PREVIEW, NO_REVIEW_RUN,
    NOT_CURRENT_RUN, NOT_IN_TASK, NOT_ON_BOARD, NOT_OWN_TASK, REVIEW_RUN_ONLY, REVIEWER,
    check_design_access, pages_branch_alias,
)
from tools.registry import registry

ALLOWED = ["raphael.agentsbridge.net", "raphael-first-project.pages.dev", "*.raphael-first-project.pages.dev"]
APEX = "raphael-first-project.pages.dev"
HOME_PAGE = "https://raphael.agentsbridge.net/"
PUBLIC_IP = "93.184.216.34"
TOOL_TASK = "design-access-test"  # the handler's task_id: only its supervisor key, never the kanban task
HEAD = "0123abcd" + "ef" * 16
ENV = ("HERMES_KANBAN_DB", "HERMES_KANBAN_BOARD", "HERMES_KANBAN_TASK", "HERMES_KANBAN_RUN_ID",
       "HERMES_KANBAN_CLAIM_LOCK")
ACTIONS = [
    {"action": "open", "url": HOME_PAGE},
    {"action": "screenshot"},
    {"action": "read_text"},
    {"action": "set_viewport", "width": 1280, "height": 800},
]


@pytest.fixture(autouse=True)
def public_dns(monkeypatch):
    """Every name resolves to a public address, at both places the address policy resolves names."""

    def getaddrinfo(host, port=None):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (PUBLIC_IP, port or 0))]

    monkeypatch.setattr(url_safety, "_getaddrinfo", getaddrinfo)
    monkeypatch.setattr(design_url_policy, "_getaddrinfo", getaddrinfo)
    for var in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy",
                "BROWSER_CDP_URL"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture(autouse=True)
def stop_design_browser():
    yield
    design_browser_tool.cleanup_design_browser(TOOL_TASK)


class FakeEndpoint:
    """A listening socket on 127.0.0.1 that counts the connections it accepts and closes each one
    at once. As ``design_browser.cdp_url`` it is the tool's only way to a browser."""

    def __init__(self) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(16)
        self._sock.settimeout(0.05)
        self.url = f"http://127.0.0.1:{self._sock.getsockname()[1]}"
        self.connections = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        while not self._stop.is_set():
            try:
                conn, _ = self._sock.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            self.connections += 1
            conn.close()

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)
        self._sock.close()


@pytest.fixture
def endpoint():
    fake = FakeEndpoint()
    yield fake
    fake.close()


def write_config(**design) -> None:
    """``config.yaml`` in the test's own HERMES_HOME (JSON is YAML)."""
    Path(os.environ["HERMES_HOME"], "config.yaml").write_text(json.dumps({"design_browser": design}),
                                                              encoding="utf-8")


class Kanban:
    """Two real boards under the root, and the dispatcher's environment for one claimed task."""

    def __init__(self, kb, root: Path, dbs: dict, monkeypatch) -> None:
        self.kb, self.root, self.dbs, self._mp = kb, root, dbs, monkeypatch

    def start(self, assignee: str, *, board: str = "design", review_by: str | None = None,
              branch: str | None = None, head: str | None = None) -> dict:
        """A task on ``board`` assigned to ``assignee``, claimed (from the review lane after
        ``request_review`` when ``review_by`` names the reviewer), with a claimed sibling task;
        then the exact environment the dispatcher injects into the task's worker. ``{id}`` in
        ``branch`` becomes the task's id."""
        kb = self.kb
        for var in ENV:  # the boards are reached by their own files, as before a worker starts
            self._mp.delenv(var, raising=False)
        with contextlib.closing(kb.connect(board=board)) as conn:
            own, sibling = (kb.create_task(conn, title=title, assignee=assignee) for title in ("own", "sibling"))
            kb.claim_task(conn, sibling)
            first = kb.claim_task(conn, own)
            assert first is not None
            if review_by:
                assert kb.request_review(conn, own, summary="ready", reviewer=review_by,
                                         expected_run_id=first.current_run_id)
                assert kb.claim_review_task(conn, own, claimer=f"{review_by}:1") is not None
            if branch:
                kb.set_branch_name(conn, own, branch.format(id=own))
            if head:
                conn.execute("UPDATE tasks SET head_commit = ? WHERE id = ?", (head, own))
                conn.commit()
            task = kb.get_task(conn, own)
        for key, value in {
            "HERMES_KANBAN_DB": str(self.dbs[board]), "HERMES_KANBAN_BOARD": board,
            "HERMES_KANBAN_TASK": own, "HERMES_KANBAN_RUN_ID": str(task.current_run_id),
            "HERMES_KANBAN_CLAIM_LOCK": task.claim_lock or "",
        }.items():
            self._mp.setenv(key, value)
        return {"own": own, "sibling": sibling, "run": task.current_run_id, "first_run": first.current_run_id}

    def raw(self, board: str, sql: str, *params):
        """An independent read of a board's own file, straight from SQLite."""
        with contextlib.closing(sqlite3.connect(self.dbs[board])) as conn:
            return conn.execute(sql, params).fetchall()


@pytest.fixture
def kanban(tmp_path, monkeypatch):
    from hermes_cli import kanban_db as kb

    root = tmp_path / "root"
    home = root / "profiles" / "designer"
    home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    for var in ("HERMES_KANBAN_HOME", "HERMES_KANBAN_ATTACHMENTS_ROOT", "HERMES_DELEGATED_CHILD_CONTEXT", *ENV):
        monkeypatch.delenv(var, raising=False)
    for board in ("design", "elsewhere"):
        kb.create_board(board)
        kb.connect(board=board).close()
    dbs = {b: kb.kanban_db_path(board=b) for b in ("design", "elsewhere")}
    assert len(set(dbs.values())) == 2 and all(db.is_file() for db in dbs.values())
    return Kanban(kb, root, dbs, monkeypatch)


def configure(endpoint: FakeEndpoint, boards=None, **extra) -> None:
    design = {"cdp_url": endpoint.url, "allowed_hosts": ALLOWED, **extra}
    if boards is not None:
        design["boards"] = boards
    write_config(**design)


def access():
    """The gate's answer, on the config section the handler loads."""
    return check_design_access(design_browser_tool._design_config())


def call(args: dict):
    return registry.dispatch("design_browser", dict(args), task_id=TOOL_TASK)


def error_of(result) -> str:
    assert isinstance(result, str), result
    payload = json.loads(result)
    assert "error" in payload and not payload.get("ok"), result
    return payload["error"]


def assert_refused(endpoint: FakeEndpoint, reason: str) -> None:
    """Refused by the gate, at the gate and through the tool for every action, with no connection."""
    answer = access()
    assert (answer.allowed, answer.reason, answer.builder_host) == (False, reason, None)
    for args in ACTIONS:
        assert error_of(call(args)) == "design browser refused: " + reason
    assert endpoint.connections == 0


def assert_reaches_browser(endpoint: FakeEndpoint, actions=ACTIONS) -> None:
    """Each call gets past the gate and the address check and connects to the fake endpoint."""
    for args in actions:
        before = endpoint.connections
        assert error_of(call(args)) == design_browser_tool.UNAVAILABLE
        assert endpoint.connections > before, args


def alias(branch: str) -> str:
    """Cloudflare Pages' branch alias (plan v2 O10), written out independently of the gate."""
    return re.sub(r"[^a-z0-9]", "-", branch.lower())[:28].strip("-")


# ── The board setting ────────────────────────────────────────────────────────


@pytest.mark.parametrize("boards", [[], None], ids=["empty", "absent"])
def test_empty_board_setting_refuses_everyone(kanban, endpoint, boards):
    kanban.start(DESIGNER)
    configure(endpoint, boards=boards)
    assert_refused(endpoint, NO_BOARDS)


def test_a_board_not_in_the_setting_is_refused(kanban, endpoint):
    kanban.start(DESIGNER)
    configure(endpoint, boards=["elsewhere"])
    assert_refused(endpoint, BOARD_NOT_LISTED)


@pytest.mark.parametrize("change", [
    ("HERMES_KANBAN_TASK", None), ("HERMES_KANBAN_RUN_ID", None), ("HERMES_KANBAN_BOARD", None),
    ("HERMES_KANBAN_RUN_ID", "not-a-number"), ("HERMES_KANBAN_RUN_ID", "0"),
], ids=["no-task", "no-run", "no-board", "run-not-an-integer", "run-zero"])
def test_outside_a_task_is_refused(kanban, endpoint, monkeypatch, change):
    kanban.start(DESIGNER)
    configure(endpoint, boards=["design"])
    var, value = change
    if value is None:
        monkeypatch.delenv(var)
    else:
        monkeypatch.setenv(var, value)
    assert_refused(endpoint, NOT_IN_TASK)


# ── Roles ────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("profile", ["raphael-planner", "raphael-business", "default", "made-up-profile"])
def test_a_non_design_assignee_is_refused_without_a_connection(kanban, endpoint, profile):
    started = kanban.start(profile)
    configure(endpoint, boards=["design"])
    assert kanban.raw("design", "SELECT profile FROM task_runs WHERE id = ?", started["run"]) == [(profile,)]
    assert_refused(endpoint, NO_DESIGN_ROLE)


def test_the_designer_on_a_listed_board_is_allowed(kanban, endpoint):
    started = kanban.start(DESIGNER)
    configure(endpoint, boards=["design"])
    answer = access()
    assert (answer.allowed, answer.role, answer.builder_host) == (True, DESIGNER, None)
    # The run row the gate saw, read independently from the board's own file.
    assert kanban.raw(
        "design",
        "SELECT r.profile, t.assignee FROM tasks t JOIN task_runs r ON r.id = t.current_run_id WHERE t.id = ?",
        started["own"]) == [(DESIGNER, DESIGNER)]
    assert int(os.environ["HERMES_KANBAN_RUN_ID"]) == started["run"]
    assert_reaches_browser(endpoint)


def test_the_designer_on_another_listed_board_is_allowed_through_that_boards_own_file(kanban, endpoint):
    kanban.start(DESIGNER, board="elsewhere")
    configure(endpoint, boards=["design", "elsewhere"])
    answer = access()
    assert (answer.allowed, answer.role, answer.builder_host) == (True, DESIGNER, None)
    assert_reaches_browser(endpoint, ACTIONS[:1])


def test_the_reviewer_on_a_run_that_is_not_a_review_run_is_refused(kanban, endpoint):
    kanban.start(REVIEWER)
    configure(endpoint, boards=["design"])
    assert_refused(endpoint, REVIEW_RUN_ONLY)


def test_the_reviewer_on_a_review_run_is_allowed(kanban, endpoint):
    started = kanban.start(BUILDER, review_by=REVIEWER)
    configure(endpoint, boards=["design"])
    assert kanban.raw("design", "SELECT profile FROM task_runs WHERE id = ?", started["run"]) == [(REVIEWER,)]
    answer = access()
    assert (answer.allowed, answer.role, answer.builder_host) == (True, REVIEWER, None)
    assert_reaches_browser(endpoint)


def test_the_builder_on_a_review_run_is_refused(kanban, endpoint):
    started = kanban.start(DESIGNER, review_by=BUILDER)  # a review is never requested from the assignee itself
    assert kanban.raw("design", "SELECT profile FROM task_runs WHERE id = ?", started["run"]) == [(BUILDER,)]
    configure(endpoint, boards=["design"])
    assert_refused(endpoint, NO_REVIEW_RUN)


def test_the_builder_on_another_task_is_refused(kanban, endpoint):
    started = kanban.start(BUILDER, branch="design-loop/gate")
    with contextlib.closing(sqlite3.connect(kanban.dbs["design"])) as conn:
        conn.execute("UPDATE tasks SET assignee = ? WHERE id = ?", (DESIGNER, started["own"]))
        conn.commit()
    configure(endpoint, boards=["design"])
    assert_refused(endpoint, NOT_OWN_TASK)


def test_the_builder_sees_only_its_own_preview(kanban, endpoint):
    started = kanban.start(BUILDER, branch="design-loop/{id}-gate")
    branch = f"design-loop/{started['own']}-gate"
    assert kanban.raw("design", "SELECT branch_name FROM tasks WHERE id = ?", started["own"]) == [(branch,)]
    configure(endpoint, boards=["design"])
    own_alias = alias(branch)
    assert re.sub(r"[^a-z0-9]", "-", started["own"].lower()) in own_alias
    assert pages_branch_alias(branch) == own_alias
    answer = access()
    assert (answer.allowed, answer.role, answer.builder_host) == (True, BUILDER, f"{own_alias}.{APEX}")
    for url in (f"https://other.{APEX}/", HOME_PAGE, f"https://{APEX}/", f"https://{HEAD[:8]}.{APEX}/"):
        assert error_of(call({"action": "open", "url": url})).startswith("address refused: "), url
    assert endpoint.connections == 0
    assert_reaches_browser(endpoint, [{"action": "open", "url": f"https://{own_alias}.{APEX}/"}])


def test_the_builder_falls_back_to_the_commit_host_when_the_alias_loses_the_task_id(kanban, endpoint):
    started = kanban.start(BUILDER, branch="design-loop/a-very-long-feature-name/gate", head=HEAD)
    assert re.sub(r"[^a-z0-9]", "-", started["own"].lower()) not in alias("design-loop/a-very-long-feature-name/gate")
    configure(endpoint, boards=["design"])
    answer = access()
    assert (answer.allowed, answer.role, answer.builder_host) == (True, BUILDER, f"{HEAD[:8]}.{APEX}")
    assert error_of(call({"action": "open", "url": f"https://other.{APEX}/"})).startswith("address refused: ")
    assert endpoint.connections == 0
    assert_reaches_browser(endpoint, [{"action": "open", "url": f"https://{HEAD[:8]}.{APEX}/"}])


@pytest.mark.parametrize("branch, head", [(None, HEAD), ("design-loop/a-very-long-feature-name/gate", None)],
                         ids=["no-branch", "alias-without-the-id-and-no-commit"])
def test_the_builder_without_a_derivable_preview_is_refused(kanban, endpoint, branch, head):
    started = kanban.start(BUILDER, branch=branch, head=head)
    assert kanban.raw("design", "SELECT branch_name FROM tasks WHERE id = ?", started["own"]) == [(branch,)]
    configure(endpoint, boards=["design"])
    assert_refused(endpoint, NO_PREVIEW)


# ── Spoofed environment ──────────────────────────────────────────────────────


def test_the_task_id_of_another_run_is_refused(kanban, endpoint, monkeypatch):
    started = kanban.start(DESIGNER)
    configure(endpoint, boards=["design"])
    monkeypatch.setenv("HERMES_KANBAN_TASK", started["sibling"])
    assert_refused(endpoint, NOT_CURRENT_RUN)


def test_a_board_that_is_not_the_pinned_database_is_refused(kanban, endpoint, monkeypatch):
    kanban.start(DESIGNER)
    configure(endpoint, boards=["design", "elsewhere"])
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "elsewhere")
    assert_refused(endpoint, NOT_ON_BOARD)


def test_a_board_that_does_not_hold_the_run_is_refused(kanban, endpoint, monkeypatch):
    started = kanban.start(DESIGNER)
    configure(endpoint, boards=["design", "elsewhere"])
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "elsewhere")
    monkeypatch.delenv("HERMES_KANBAN_DB")
    # Independently: "elsewhere"'s own file holds no such run of this task.
    assert kanban.raw("elsewhere", "SELECT id FROM task_runs WHERE id = ? AND task_id = ?",
                      started["run"], started["own"]) == []
    assert_refused(endpoint, NOT_CURRENT_RUN)


@pytest.mark.parametrize("which", ["old-run", "no-such-run"])
def test_a_run_that_is_not_the_tasks_current_run_is_refused(kanban, endpoint, monkeypatch, which):
    started = kanban.start(DESIGNER, review_by=REVIEWER)  # the designer's run is now an old run
    assert started["first_run"] != started["run"]
    configure(endpoint, boards=["design"])
    run = started["first_run"] if which == "old-run" else started["run"] + 1000
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run))
    assert kanban.raw("design", "SELECT current_run_id FROM tasks WHERE id = ?", started["own"]) == [(started["run"],)]
    assert_refused(endpoint, NOT_CURRENT_RUN)


# ── The all alias and check_fn ───────────────────────────────────────────────


def test_the_all_alias_shows_the_tool_but_every_call_is_refused(kanban, endpoint):
    assert "design_browser" in toolsets.resolve_toolset("all")
    kanban.start("raphael-planner")
    configure(endpoint, boards=["design"])
    assert_refused(endpoint, NO_DESIGN_ROLE)


def test_check_fn_answers_only_whether_the_cdp_url_is_set(kanban, endpoint):
    kanban.start("raphael-planner")
    configure(endpoint, boards=[])
    assert design_browser_tool.check_design_browser_requirements() is True
    write_config(cdp_url="", allowed_hosts=ALLOWED, boards=["design"])
    assert design_browser_tool.check_design_browser_requirements() is False
    assert endpoint.connections == 0


# ── The builder's own preview, with other design tasks on the same browser ──


def test_the_builders_guard_holds_only_its_own_contexts_requests_to_its_preview():
    """Recorded CDP calls, no browser. The builder's guard pauses the requests of the targets in
    its own browser context on their own sessions, before they run, and answers those with its
    own preview host; a request of another task, paused browser-wide, it answers by
    ``design_browser.allowed_hosts`` alone, as every design guard did before."""
    own, other = f"own-gate.{APEX}", f"other-gate.{APEX}"
    calls = []

    async def cdp(method, params=None, *, session_id=None, timeout=None):
        calls.append((method, session_id, (params or {}).get("requestId")))
        return {}

    key = design_browser_tool.supervisor_key("scoping-test")
    guard = design_browser_tool._guard(key, ALLOWED, own)
    try:
        assert (guard.policy, guard.builder_host) == (ALLOWED, None)  # the browser-wide decision is unchanged
        guard.scope("own-context")

        def paused(request_id, url, frame, kind="Document"):
            return {"requestId": request_id, "resourceType": kind, "frameId": frame,
                    "request": {"method": "GET", "url": url, "headers": {}}}

        async def scenario():
            for sid, context in (("own-page", "own-context"), ("other-page", "other-context")):
                await guard.guard_target(cdp, {"sessionId": sid, "waitingForDebugger": True, "targetInfo": {
                    "type": "page", "targetId": sid + "-target", "browserContextId": context}}, None)
            guard.watch("own-page-target")
            for event, sid in ((paused("hop", f"https://{other}/", "own-page-target"), "own-page"),
                               (paused("own", f"https://{own}/app.js", "own-page-target", "Script"), "own-page"),
                               (paused("designer", f"https://{other}/shot.png", "other-page-target", "Image"), None),
                               (paused("outside", "https://example.com/", "other-page-target"), None)):
                assert guard.on_event(cdp, "Fetch.requestPaused", event, sid) is True
            for _ in range(20):
                await asyncio.sleep(0)

        asyncio.run(scenario())
        enabled = [(m, sid) for m, sid, _ in calls if m == "Fetch.enable"]
        assert enabled == [("Fetch.enable", "own-page")]  # never on another task's target
        resumed = calls.index(("Runtime.runIfWaitingForDebugger", "own-page", None))
        assert calls.index(("Fetch.enable", "own-page", None)) < resumed
        answers = {rid: (m, sid) for m, sid, rid in calls if m in ("Fetch.continueRequest", "Fetch.failRequest")}
        assert answers == {"hop": ("Fetch.failRequest", "own-page"), "own": ("Fetch.continueRequest", "own-page"),
                           "designer": ("Fetch.continueRequest", None), "outside": ("Fetch.failRequest", None)}
        reason = design_url_policy.design_url_refusal(f"https://{other}/", ALLOWED, builder_host=own)
        assert reason and guard.unwatch("own-page-target") == (other, reason)
    finally:
        design_browser_tool._forget_guard(key)


class _Previews(http.server.BaseHTTPRequestHandler):
    """Every design host on one local TLS server, told apart by the Host header. ``hits`` counts
    each request that reached it, as (host, path)."""

    hits: list = []
    pages: dict = {}  # (host, path) -> HTML
    redirects: dict = {}  # (host, path) -> address it redirects to

    def do_GET(self):
        host, path = (self.headers.get("Host") or "").split(":")[0], self.path.split("?")[0]
        _Previews.hits.append((host, path))
        location = _Previews.redirects.get((host, path))
        body = b"" if location else _Previews.pages.get((host, path), f"<title>{path}</title>").encode()
        self.send_response(302 if location else 200)
        if location:
            self.send_header("Location", location)
        self.send_header("Content-Type", "text/javascript" if path.endswith(".js") else "text/html")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        pass


@pytest.fixture
def design_chromium(tmp_path):
    """Headless Chromium as the design browser runs, every design host mapped to one local TLS
    server (self-signed, accepted); yields its DevTools address. Only the names are mapped, so
    the addresses keep the default port and the tool's own address policy runs on them, with
    only DNS stubbed (``public_dns``)."""
    chrome = shutil.which("google-chrome") or shutil.which("chromium")
    if not chrome or not shutil.which("openssl"):
        pytest.skip("Chromium or openssl not installed")
    cert, key = tmp_path / "cert.pem", tmp_path / "key.pem"
    subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-keyout", str(key), "-out",
                    str(cert), "-days", "1", "-subj", f"/CN={APEX}"],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    _Previews.hits, _Previews.pages, _Previews.redirects = [], {}, {}
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Previews)
    tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    tls.load_cert_chain(cert, key)
    server.socket = tls.wrap_socket(server.socket, server_side=True)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    site = f"127.0.0.1:{server.server_address[1]}"
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        address = f"http://127.0.0.1:{probe.getsockname()[1]}"
    proc = subprocess.Popen(
        [chrome, f"--remote-debugging-port={address.rsplit(':', 1)[1]}", f"--user-data-dir={tmp_path / 'profile'}",
         "--no-first-run", "--no-default-browser-check", "--headless=new", "--disable-gpu",
         "--block-new-web-contents", "--enable-automation", "--ignore-certificate-errors",
         f"--host-resolver-rules=MAP *.{APEX} {site}, MAP {APEX} {site}, MAP raphael.agentsbridge.net {site}"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env={**os.environ, "HOME": str(tmp_path)})
    try:
        deadline = time.monotonic() + 15
        while True:
            try:
                with urllib.request.urlopen(address + "/json/version", timeout=1):
                    break
            except Exception:
                if time.monotonic() > deadline:
                    pytest.skip("Chromium didn't expose CDP in time")
                time.sleep(0.25)
        yield address
    finally:
        proc.kill()
        proc.wait(timeout=5)
        server.shutdown()
        server.server_close()


def opened(result) -> dict:
    """An ``open`` result's JSON, from inside the untrusted-data wrapper when it has one."""
    assert isinstance(result, str), result
    if result.startswith("{"):
        return json.loads(result)
    return json.loads(result.split("\n\n", 1)[1].rsplit("\n</untrusted_tool_result>", 1)[0])


@pytest.mark.integration
@pytest.mark.skipif(os.environ.get("HERMES_E2E_BROWSER", "").strip() != "1",
                    reason="real-browser E2E: set HERMES_E2E_BROWSER=1 to opt in")
@pytest.mark.skipif(not shutil.which("google-chrome") and not shutil.which("chromium"),
                    reason="Chrome/Chromium not installed")
def test_live_builders_keep_to_their_own_previews_and_leave_the_designer_its_allow_list(
        kanban, design_chromium, monkeypatch):
    """Two builders (two tasks, two branches, two preview hosts) and the designer, each in its
    own worker's environment and under its own supervisor key, live together on one Chromium.
    Each page the gate permits loads from the server; nothing of the other builder's preview
    reaches it from a builder's page (direct, redirect hop, image, frame, fetch, worker); the
    designer still loads what ``design_browser.allowed_hosts`` allows, a builder's preview too."""
    workers = {}
    for name, assignee in (("first", BUILDER), ("second", BUILDER), ("designer", DESIGNER)):
        started = kanban.start(assignee, branch="design-loop/{id}-gate" if assignee == BUILDER else None)
        workers[name] = {"task": started["own"], "env": {var: os.environ[var] for var in ENV}}
    write_config(cdp_url=design_chromium, allowed_hosts=ALLOWED, boards=["design"])

    def use(name):
        for var, value in workers[name]["env"].items():
            monkeypatch.setenv(var, value)

    def open_as(name, url):
        use(name)
        return opened(registry.dispatch("design_browser", {"action": "open", "url": url},
                                        task_id=workers[name]["task"]))

    preview = {}
    for name in ("first", "second"):
        ((branch,),) = kanban.raw("design", "SELECT branch_name FROM tasks WHERE id = ?", workers[name]["task"])
        preview[name] = f"{alias(branch)}.{APEX}"
        use(name)
        assert access().builder_host == preview[name]
    first, second, home = preview["first"], preview["second"], "raphael.agentsbridge.net"
    assert first != second and len({w["task"] for w in workers.values()}) == 3
    _Previews.redirects[(first, "/to-second")] = f"https://{second}/redirected"
    _Previews.pages[(first, "/embeds-second")] = (
        f"<title>embeds</title><img src='https://{second}/image.png'><iframe src='https://{second}/frame'></iframe>"
        f"<img src='https://{home}/from-first.png'>"
        f"<script>fetch('https://{second}/fetched').catch(() => 0); new Worker('/worker.js');</script>")
    _Previews.pages[(first, "/worker.js")] = f"fetch('https://{second}/worker-fetched').catch(() => 0);"
    _Previews.pages[(home, "/designer")] = f"<title>designer</title><img src='https://{first}/shown.png'>"
    try:
        for name, url in (("first", f"https://{first}/own"), ("second", f"https://{second}/own"),
                          ("designer", f"https://{home}/designer"), ("designer", f"https://{second}/as-designer")):
            data = open_as(name, url)
            assert data["ok"] is True, (name, url, data)
        assert {(first, "/own"), (second, "/own"), (home, "/designer"), (first, "/shown.png"),
                (second, "/as-designer")} <= set(_Previews.hits)

        assert open_as("first", f"https://{second}/direct")["error"].startswith("address refused: ")
        hop = open_as("first", f"https://{first}/to-second")
        assert (hop["ok"], hop["blocked_host"]) == (False, second) and hop["reason"], hop
        assert open_as("first", f"https://{first}/embeds-second")["ok"] is True
        time.sleep(1.5)  # the page's fetch and worker have had their time
        assert (first, "/to-second") in _Previews.hits and (first, "/worker.js") in _Previews.hits
        assert [path for host, path in _Previews.hits if host == second] == ["/own", "/as-designer"]
        assert (home, "/from-first.png") not in _Previews.hits  # allowed for others, not the builder's own

        shown = _Previews.hits.count((first, "/shown.png"))
        assert open_as("designer", f"https://{home}/designer")["ok"] is True  # both builders still live
        assert _Previews.hits.count((first, "/shown.png")) == shown + 1
        assert open_as("second", f"https://{second}/own")["ok"] is True
        assert _Previews.hits.count((second, "/own")) == 2
    finally:
        for worker in workers.values():
            design_browser_tool.cleanup_design_browser(worker["task"])
