"""Navigation guard in the CDP supervisor (design-loop plan v2, B2).

The default tests run a real ``CDPSupervisor`` against a fake CDP socket (an in-process WebSocket
server that answers every command and records it) and use the real allow-list decision from
``tools.design_url_policy`` with DNS stubbed to a public address.

``TestRealChromium`` drives headless Chromium. Like ``tests/tools/test_browser_supervisor.py`` it
is opt-in twice over: ``@pytest.mark.integration`` and ``HERMES_E2E_BROWSER=1``. Its pages come
from a server on 127.0.0.1, which the design allow-list always refuses, so those tests swap the
decision for one that allows only that server's ``/ok`` paths; they check what Chromium does with
the guard's answers, not the allow-list itself. Run them with:

    HERMES_E2E_BROWSER=1 scripts/run_tests.sh -m integration tests/tools/test_browser_navguard.py
"""

from __future__ import annotations

import asyncio
import http.server
import json
import os
import shutil
import socket
import subprocess
import tempfile
import threading
import time
import urllib.request

import pytest

from tools import browser_supervisor_navguard as navguard
from tools import design_url_policy, url_safety
from tools.browser_supervisor import CDPSupervisor, _SupervisorRegistry

ALLOWED = ["raphael.agentsbridge.net", "raphael-first-project.pages.dev", "*.raphael-first-project.pages.dev"]
PUBLIC_IP = "93.184.216.34"


@pytest.fixture(autouse=True)
def public_dns(monkeypatch):
    """Every name resolves to a public address, at both places the policy resolves names."""

    def getaddrinfo(host, port=None):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (PUBLIC_IP, port or 0))]

    monkeypatch.setattr(url_safety, "_getaddrinfo", getaddrinfo)
    monkeypatch.setattr(design_url_policy, "_getaddrinfo", getaddrinfo)
    for var in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.delenv(var, raising=False)


# ── Fake CDP socket ──────────────────────────────────────────────────────────


class FakeCDP:
    """A CDP endpoint that answers every command with ``{}`` (or a canned result / error) and
    records it. ``push`` sends an event to the connected supervisor."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.errors: set[tuple[str, str | None]] = set()  # (method, sessionId) answered with an error
        self.results = {
            "Target.getTargets": {"targetInfos": [{"targetId": "T1", "type": "page", "url": "about:blank"}]},
            "Target.attachToTarget": {"sessionId": "S1"},
        }
        self._conn = None
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()
        self._ready.wait(5)

    def _serve(self) -> None:
        import websockets

        async def handler(conn):
            self._conn = conn
            async for raw in conn:
                msg = json.loads(raw)
                self.calls.append(msg)
                reply = {"id": msg["id"]}
                if (msg["method"], msg.get("sessionId")) in self.errors:
                    reply["error"] = {"code": -32000, "message": "fake failure"}
                else:
                    reply["result"] = self.results.get(msg["method"], {})
                await conn.send(json.dumps(reply))

        async def main():
            self._server = await websockets.serve(handler, "127.0.0.1", 0)
            self.url = f"ws://127.0.0.1:{self._server.sockets[0].getsockname()[1]}/devtools/browser/fake"
            self._ready.set()
            await self._server.wait_closed()

        self._loop.run_until_complete(main())

    def push(self, method: str, params: dict, session_id: str | None = None) -> None:
        event = {"method": method, "params": params}
        if session_id:
            event["sessionId"] = session_id
        asyncio.run_coroutine_threadsafe(self._conn.send(json.dumps(event)), self._loop).result(5)

    def wait_for(self, predicate, timeout: float = 5.0, start: int = 0) -> dict:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            for call in self.calls[start:]:
                if predicate(call):
                    return call
            time.sleep(0.02)
        raise AssertionError("expected CDP call never arrived")

    def methods(self, session_id: str | None = None) -> list[str]:
        return [c["method"] for c in self.calls if c.get("sessionId") == session_id]

    def close(self) -> None:
        self._loop.call_soon_threadsafe(self._server.close)
        self._thread.join(5)


@pytest.fixture
def fake_cdp():
    fake = FakeCDP()
    yield fake
    fake.close()


@pytest.fixture
def supervise(fake_cdp):
    """Start a supervisor on the fake socket; stopped after the test."""
    started = []

    def start(**kwargs) -> CDPSupervisor:
        supervisor = CDPSupervisor(task_id="navguard-test", cdp_url=fake_cdp.url, **kwargs)
        started.append(supervisor)
        supervisor.start(timeout=10)
        return supervisor

    yield start
    for supervisor in started:
        supervisor.stop()


def _paused(fake: FakeCDP, request_id: str, url: str, *, method: str = "GET", headers=None,
            resource_type: str = "Document") -> str:
    """Pause one request at the browser level and return the guard's answer to it."""
    start = len(fake.calls)
    fake.push("Fetch.requestPaused", {
        "requestId": request_id, "resourceType": resource_type,
        "request": {"url": url, "method": method, "headers": headers or {}},
    })
    answer = fake.wait_for(lambda c: c["method"].startswith("Fetch.") and c.get("params", {}).get("requestId") == request_id,
                           start=start)
    return answer["method"]


def _guard(**kwargs) -> navguard.NavigationGuard:
    return navguard.NavigationGuard(ALLOWED, **kwargs)


# ── Guard off unless asked for ───────────────────────────────────────────────


def test_guard_is_off_unless_asked_for(fake_cdp, supervise):
    supervise()
    assert "Fetch.enable" not in fake_cdp.methods(None)
    assert "Browser.setDownloadBehavior" not in fake_cdp.methods(None)
    auto_attach = [c for c in fake_cdp.calls if c["method"] == "Target.setAutoAttach"]
    assert auto_attach and not any(c["params"]["waitForDebuggerOnStart"] for c in auto_attach)


def test_registry_starts_the_guard_only_when_given_one(fake_cdp):
    registry = _SupervisorRegistry()
    try:
        plain = registry.get_or_start(task_id="plain", cdp_url=fake_cdp.url)
        assert plain.navigation_guard is None
        guard = _guard()
        guarded = registry.get_or_start(task_id="guarded", cdp_url=fake_cdp.url, navigation_guard=guard)
        assert guarded.navigation_guard is guard
    finally:
        registry.stop_all()


def test_registry_never_swaps_a_guarded_supervisor_for_an_unguarded_one(fake_cdp):
    registry = _SupervisorRegistry()
    try:
        plain = registry.get_or_start(task_id="t", cdp_url=fake_cdp.url)
        guard = _guard()
        guarded = registry.get_or_start(task_id="t", cdp_url=fake_cdp.url, navigation_guard=guard)
        assert guarded is not plain and guarded.navigation_guard is guard
        # Callers that don't ask for a guard (the browser tools) keep the guarded one.
        assert registry.get_or_start(task_id="t", cdp_url=fake_cdp.url) is guarded
    finally:
        registry.stop_all()


# ── Installs on attach and on every new target before it resumes ─────────────


def test_guard_installs_before_start_returns(fake_cdp, supervise):
    supervise(navigation_guard=_guard())
    root = {c["method"]: c.get("params", {}) for c in fake_cdp.calls if "sessionId" not in c}
    assert root["Fetch.enable"]["patterns"] == [{"urlPattern": "*", "requestStage": "Request"}]
    assert root["Browser.setDownloadBehavior"] == {"behavior": "deny"}
    denied = {c["params"]["permission"]["name"] for c in fake_cdp.calls if c["method"] == "Browser.setPermission"}
    assert {"clipboard-read", "clipboard-write", "geolocation", "notifications"} <= denied
    assert all(c["params"]["setting"] == "denied" for c in fake_cdp.calls if c["method"] == "Browser.setPermission")
    assert root["Target.setAutoAttach"]["waitForDebuggerOnStart"] is True
    assert {"Page.setInterceptFileChooserDialog", "Network.enable"} <= set(fake_cdp.methods("S1"))


def test_new_target_is_guarded_before_it_resumes(fake_cdp, supervise):
    supervise(navigation_guard=_guard())
    fake_cdp.push("Target.attachedToTarget", {
        "sessionId": "S2", "waitingForDebugger": True,
        "targetInfo": {"targetId": "T2", "type": "page", "url": "about:blank"},
    })
    fake_cdp.wait_for(lambda c: c["method"] == "Runtime.runIfWaitingForDebugger" and c.get("sessionId") == "S2")
    methods = fake_cdp.methods("S2")
    resume = methods.index("Runtime.runIfWaitingForDebugger")
    assert methods.index("Page.setInterceptFileChooserDialog") < resume
    assert methods.index("Network.enable") < resume


def test_new_target_that_cannot_be_guarded_is_closed_not_resumed(fake_cdp, supervise):
    supervise(navigation_guard=_guard())
    fake_cdp.errors.add(("Page.setInterceptFileChooserDialog", "S3"))
    fake_cdp.push("Target.attachedToTarget", {
        "sessionId": "S3", "waitingForDebugger": True,
        "targetInfo": {"targetId": "T3", "type": "page", "url": "about:blank"},
    })
    closed = fake_cdp.wait_for(lambda c: c["method"] == "Target.closeTarget")
    assert closed["params"] == {"targetId": "T3"}
    assert "Runtime.runIfWaitingForDebugger" not in fake_cdp.methods("S3")


def test_start_fails_when_the_guard_cannot_install(fake_cdp, supervise):
    fake_cdp.errors.add(("Fetch.enable", None))
    with pytest.raises(RuntimeError):
        supervise(navigation_guard=_guard())


# ── Requests ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("url", [
    "https://raphael.agentsbridge.net/",
    "https://raphael-first-project.pages.dev/index.html",
    "https://abc123.raphael-first-project.pages.dev/styles.css",
])
def test_allowed_address_opens(fake_cdp, supervise, url):
    supervise(navigation_guard=_guard())
    assert _paused(fake_cdp, "r1", url) == "Fetch.continueRequest"


@pytest.mark.parametrize("url", [
    "https://evil.example/",
    "http://raphael.agentsbridge.net/",
    "https://a.b.raphael-first-project.pages.dev/",
    "https://static.cloudflareinsights.com/beacon.min.js",
    "https://raphael.agentsbridge.net/cdn-cgi/rum",
    "https://127.0.0.1/",
    "data:text/html,hi",
])
def test_refused_address_is_failed(fake_cdp, supervise, url):
    supervise(navigation_guard=_guard())
    assert _paused(fake_cdp, "r1", url) == "Fetch.failRequest"
    failed = [c for c in fake_cdp.calls if c["method"] == "Fetch.failRequest"][-1]
    assert failed["params"] == {"requestId": "r1", "errorReason": "BlockedByClient"}


def test_redirect_hop_to_a_refused_address_is_failed(fake_cdp, supervise):
    """Fetch pauses every redirect hop again at the request stage, with the hop's URL."""
    supervise(navigation_guard=_guard())
    assert _paused(fake_cdp, "r1", "https://raphael.agentsbridge.net/go") == "Fetch.continueRequest"
    assert _paused(fake_cdp, "r1", "https://evil.example/landing") == "Fetch.failRequest"


def test_builder_is_narrowed_to_its_own_preview(fake_cdp, supervise):
    supervise(navigation_guard=_guard(builder_host="task-42.raphael-first-project.pages.dev"))
    assert _paused(fake_cdp, "r1", "https://task-42.raphael-first-project.pages.dev/") == "Fetch.continueRequest"
    assert _paused(fake_cdp, "r2", "https://raphael.agentsbridge.net/") == "Fetch.failRequest"


@pytest.mark.parametrize("method", ["POST", "PUT", "DELETE", "OPTIONS", "HEAD"])
def test_non_get_method_is_failed(fake_cdp, supervise, method):
    supervise(navigation_guard=_guard())
    assert _paused(fake_cdp, "r1", "https://raphael.agentsbridge.net/form", method=method) == "Fetch.failRequest"


def test_service_worker_registration_is_failed(fake_cdp, supervise):
    supervise(navigation_guard=_guard())
    answer = _paused(fake_cdp, "r1", "https://raphael.agentsbridge.net/sw.js",
                     headers={"Service-Worker": "script"}, resource_type="Other")
    assert answer == "Fetch.failRequest"


def test_policy_error_fails_closed(fake_cdp, supervise, monkeypatch):
    def broken(*_args, **_kwargs):
        raise RuntimeError("policy blew up")

    monkeypatch.setattr(navguard, "design_url_refusal", broken)
    supervise(navigation_guard=_guard())
    assert _paused(fake_cdp, "r1", "https://raphael.agentsbridge.net/") == "Fetch.failRequest"


def test_websocket_closes_its_page(fake_cdp, supervise):
    """CDP Fetch never sees a WebSocket handshake (see TestRealChromium), so a page that opens a
    socket is closed as soon as Chromium reports it."""
    supervise(navigation_guard=_guard())
    fake_cdp.push("Network.webSocketCreated", {"requestId": "w1", "url": "wss://evil.example/s"}, session_id="S1")
    closed = fake_cdp.wait_for(lambda c: c["method"] == "Target.closeTarget")
    assert closed["params"] == {"targetId": "T1"}


def test_file_chooser_is_left_unanswered_and_dialog_bridge_still_works(fake_cdp, supervise):
    supervisor = supervise(navigation_guard=_guard())
    fake_cdp.push("Page.fileChooserOpened", {"frameId": "F1", "mode": "selectSingle"}, session_id="S1")
    fake_cdp.push("Fetch.requestPaused", {
        "requestId": "b1", "request": {"url": "http://hermes-dialog-bridge.invalid/?kind=alert&message=hi", "method": "GET"},
    }, session_id="S1")
    deadline = time.monotonic() + 5
    while not supervisor.snapshot().pending_dialogs and time.monotonic() < deadline:
        time.sleep(0.02)
    assert [d.message for d in supervisor.snapshot().pending_dialogs] == ["hi"]
    assert not [c for c in fake_cdp.calls if c["method"] in ("DOM.setFileInputFiles", "Fetch.failRequest")]


# ── Real Chromium ────────────────────────────────────────────────────────────


def _find_chrome() -> str:
    for candidate in ("google-chrome", "chromium", "chromium-browser"):
        path = shutil.which(candidate)
        if path:
            return path
    pytest.skip("no Chrome binary found")


class _Site(http.server.BaseHTTPRequestHandler):
    """``/ok*`` pages, ``/redirect`` (302 to ``/refused``), ``/refused`` and a page that opens a
    WebSocket to ``ws_port``. Records each path it is asked for."""

    seen: list[str] = []
    ws_port = 0

    def do_GET(self):
        type(self).seen.append(self.path)
        if self.path.startswith("/redirect"):
            self.send_response(302)
            self.send_header("Location", "/refused")
            self.end_headers()
            return
        body = "<title>ok</title><img src='/refused.png'>"
        if self.path.startswith("/ok-ws"):
            body += f"<script>new WebSocket('ws://127.0.0.1:{self.ws_port}/')</script>"
        if self.path.startswith("/ok-popup"):
            body += "<script>window.open('/refused-popup')</script>"
        data = body.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *_args):
        pass


@pytest.mark.integration
@pytest.mark.skipif(os.environ.get("HERMES_E2E_BROWSER", "").strip() != "1",
                    reason="real-browser E2E: set HERMES_E2E_BROWSER=1 to opt in")
@pytest.mark.skipif(not shutil.which("google-chrome") and not shutil.which("chromium"),
                    reason="Chrome/Chromium not installed")
class TestRealChromium:
    @pytest.fixture
    def site(self):
        handshakes: list[str] = []

        def ws_server():
            sock = socket.socket()
            sock.bind(("127.0.0.1", 0))
            sock.listen()
            _Site.ws_port = sock.getsockname()[1]
            ready.set()
            while True:
                try:
                    conn, _ = sock.accept()
                except OSError:
                    return
                handshakes.append(conn.recv(4096).decode(errors="replace").split("\r\n")[0])
                conn.close()

        ready = threading.Event()
        threading.Thread(target=ws_server, daemon=True).start()
        ready.wait(5)
        _Site.seen = []
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Site)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        yield f"http://127.0.0.1:{server.server_address[1]}", _Site.seen, handshakes
        server.shutdown()

    @pytest.fixture
    def chrome(self):
        profile = tempfile.mkdtemp(prefix="hermes-navguard-test-")
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        proc = subprocess.Popen(
            [_find_chrome(), f"--remote-debugging-port={port}", f"--user-data-dir={profile}",
             "--no-first-run", "--no-default-browser-check", "--headless=new", "--disable-gpu"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        ws_url = None
        deadline = time.monotonic() + 15
        while ws_url is None and time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/json/version", timeout=1) as r:
                    ws_url = json.loads(r.read().decode())["webSocketDebuggerUrl"]
            except Exception:
                time.sleep(0.25)
        try:
            if ws_url is None:
                pytest.skip("Chrome didn't expose CDP in time")
            yield ws_url
        finally:
            proc.kill()
            proc.wait(timeout=5)
            shutil.rmtree(profile, ignore_errors=True)

    @pytest.fixture
    def asked(self):
        """Every address the guard asked the (swapped) decision about."""
        return []

    @pytest.fixture
    def guarded(self, chrome, site, asked, monkeypatch):
        """A guarded supervisor whose decision allows only GETs of the local site's ``/ok`` paths."""
        base = site[0]

        def local_refusal(url, _policy, *, builder_host=None):
            asked.append(url)
            return None if url.startswith(base + "/ok") else "not on the test allow-list"

        monkeypatch.setattr(navguard, "design_url_refusal", local_refusal)
        supervisor = CDPSupervisor(task_id="navguard-e2e", cdp_url=chrome,
                                   navigation_guard=navguard.NavigationGuard([]))
        supervisor.start(timeout=15)
        yield supervisor
        supervisor.stop()

    @staticmethod
    def _navigate(supervisor, url: str) -> dict:
        out = supervisor.evaluate_runtime(f"location.href = {json.dumps(url)}; true")
        time.sleep(1.5)
        return out

    def test_allowed_page_opens_and_refused_subresource_is_blocked(self, guarded, site, asked):
        base, seen, _ = site
        self._navigate(guarded, base + "/ok")
        assert guarded.evaluate_runtime("document.title")["result"] == "ok"
        assert "/ok" in seen and "/refused.png" not in seen
        assert base + "/refused.png" in asked

    def test_refused_page_is_blocked(self, guarded, site, asked):
        base, seen, _ = site
        self._navigate(guarded, base + "/ok")
        self._navigate(guarded, base + "/refused")
        assert "/refused" not in seen and base + "/refused" in asked
        assert guarded.evaluate_runtime("location.href")["result"] != base + "/refused"

    def test_redirect_to_refused_page_is_blocked(self, guarded, site, asked, monkeypatch):
        base, seen, _ = site
        self._navigate(guarded, base + "/ok")
        # /redirect itself is allowed here, so only its hop to /refused is left to the guard.
        local = navguard.design_url_refusal
        monkeypatch.setattr(navguard, "design_url_refusal",
                            lambda url, *a, **k: None if url == base + "/redirect" else local(url, *a, **k))
        self._navigate(guarded, base + "/redirect")
        assert "/redirect" in seen and "/refused" not in seen
        assert base + "/refused" in asked

    def test_popup_to_refused_page_is_blocked(self, guarded, site, asked):
        base, seen, _ = site
        self._navigate(guarded, base + "/ok-popup")
        assert "/ok-popup" in seen and "/refused-popup" not in seen
        assert base + "/refused-popup" in asked

    def test_websocket_is_not_seen_by_fetch_and_its_page_is_closed(self, guarded, site, asked, chrome):
        """O3: CDP Fetch does not pause WebSocket handshakes, so the guard cannot fail them like
        other requests; it closes the page that opened the socket instead."""
        base, _, _ = site
        self._navigate(guarded, base + "/ok")
        pages = chrome.replace("ws://", "http://").split("/devtools")[0] + "/json/list"

        def page_ids():
            with urllib.request.urlopen(pages, timeout=5) as r:
                return {t["id"] for t in json.load(r) if t["type"] == "page"}

        before = page_ids()
        guarded.evaluate_runtime(f"new WebSocket('ws://127.0.0.1:{_Site.ws_port}/'); true")
        time.sleep(1.5)
        assert not [u for u in asked if f":{_Site.ws_port}" in u]  # Fetch never paused the handshake
        assert not (before & page_ids())
