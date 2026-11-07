"""Public ``AIAgent.close()`` ends the design browser of the agent's own task, and no other task's.

The real-Chromium tests open two design tasks on one headless Chromium, through the production tool
dispatcher, with the pages served over TLS by the design browser safety tests' site
(``tests/tools/test_design_browser_safety.py``, whose helpers this reuses). Each task's page fetches
an allowed path every 200 ms, tagged with its task, so a task's background requests can be counted.
They are opt-in twice over, like the safety file's: ``@pytest.mark.integration`` and
``HERMES_E2E_BROWSER=1``. Run them with:

    HERMES_E2E_BROWSER=1 scripts/run_tests.sh -m integration tests/agent/test_design_browser_close.py

The default test closes an agent that never used the design browser.
"""

from __future__ import annotations

import http.server
import os
import shutil
import ssl
import subprocess
import threading
import time

import pytest

import run_agent
from model_tools import handle_function_call
from run_agent import AIAgent
from tests.tools import test_design_browser_safety as safety
from tests.tools.test_design_browser_safety import worker  # noqa: F401  (a fixture)
from tools import design_browser_tool
from tools import process_registry as processes
from tools.browser_supervisor import SUPERVISOR_REGISTRY
from tools.computer_use import tool as computer_use

TASK_A, TASK_B = "design-close-a", "design-close-b"
TIMER_PAGE = ("<title>timer</title><script>const t = new URLSearchParams(location.search).get('t');"
              "setInterval(() => fetch('/ok-alive?t=' + t), 200)</script>")


def new_agent(session_id: str) -> AIAgent:
    """A fully initialized agent whose session, and so whose task, is ``session_id``."""
    return AIAgent(api_key="test", base_url="http://127.0.0.1:9/v1", provider="openai-compat", model="test",
                   enabled_toolsets=["design_vision"], quiet_mode=True, skip_context_files=True,
                   skip_memory=True, skip_background_review=True, session_id=session_id)


def call(args: dict, task_id: str) -> str:
    """The tool through the production dispatcher, as the agent's tool loop calls it."""
    return handle_function_call("design_browser", args, task_id=task_id,
                                enabled_tools=["design_browser"], enabled_toolsets=["design_vision"])


def alive(seen: list[str], task_id: str) -> int:
    """How many background requests the task's page has made."""
    return seen.count(f"GET /ok-alive?t={task_id}")


def background_requests(seen: list[str]) -> tuple[dict[str, int], dict[str, int]]:
    """Each task's background requests at the start and at the end of a 1.5 s window."""
    time.sleep(0.3)  # a request already on its way may still land
    before = {task: alive(seen, task) for task in (TASK_A, TASK_B)}
    time.sleep(1.5)
    return before, {task: alive(seen, task) for task in (TASK_A, TASK_B)}


def contexts(key: str) -> list[str]:
    """Every incognito context in the browser, asked through that task's supervisor."""
    return design_browser_tool._cdp(SUPERVISOR_REGISTRY.get(key), "Target.getBrowserContexts")["browserContextIds"]


def test_an_agent_that_never_used_the_design_browser_closes_as_before(monkeypatch):
    """The other teardown steps run with the agent's task id; the design browser, though configured,
    is never contacted."""
    task = "design-close-unused"
    key = design_browser_tool.supervisor_key(task)
    discovery = safety.Discovery(None)  # stands in for the design browser: counts every request
    calls: list[tuple] = []
    try:
        safety.write_config({"cdp_url": discovery.url, "allowed_hosts": safety.ALLOWED})
        agent = new_agent(task)
        for module, name in ((run_agent, "cleanup_vm"), (run_agent, "cleanup_browser"),
                             (computer_use, "release_computer_use_session")):
            monkeypatch.setattr(module, name, lambda task_id, name=name: calls.append((name, task_id)))
        monkeypatch.setattr(processes.process_registry, "list_sessions",
                            lambda *_args, **_kwargs: calls.append(("list_sessions",)) or [])
        agent.close()
    finally:
        discovery.close()
    assert sorted(calls) == sorted([("list_sessions",), ("cleanup_vm", task), ("cleanup_browser", task),
                                    ("release_computer_use_session", task)])
    assert discovery.hits == []
    assert SUPERVISOR_REGISTRY.get(key) is None
    assert key not in design_browser_tool._SESSIONS and key not in design_browser_tool._GUARDS


# ── Real Chromium ────────────────────────────────────────────────────────────


@pytest.fixture
def tls_site(tmp_path, monkeypatch):
    """The safety tests' pages, and the timer page, over TLS (self-signed; Chromium is told to accept
    it): two tasks' guards share one browser only on https pages. Yields the base address and every
    request the site got."""
    if not shutil.which("openssl"):
        pytest.skip("openssl not installed")
    cert, key = tmp_path / "cert.pem", tmp_path / "key.pem"
    subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-keyout", str(key), "-out",
                    str(cert), "-days", "1", "-subj", "/CN=127.0.0.1"],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    seen: list[str] = []
    monkeypatch.setattr(safety._Site, "seen", seen)
    monkeypatch.setitem(safety._Site.PAGES, "/ok-timer", TIMER_PAGE)
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), safety._Site)
    tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    tls.load_cert_chain(cert, key)
    server.socket = tls.wrap_socket(server.socket, server_side=True)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"https://127.0.0.1:{server.server_address[1]}", seen
    server.shutdown()
    server.server_close()


@pytest.fixture
def two_tasks(worker, tls_site, monkeypatch):
    """Tasks A and B, each with its timer page open in its own context on one real Chromium, and a
    fully initialized agent for task A. Both tasks' design browsers are ended afterwards."""
    base, seen = tls_site
    safety.TestRealChromium._allow_only(base, monkeypatch)
    launch = safety.TestRealChromium._launch(*safety.DESIGN_BROWSER_SWITCHES, "--ignore-certificate-errors")
    with launch as (address, _home):
        agent = new_agent(TASK_A)
        try:
            safety.write_config({"cdp_url": address, "allowed_hosts": ["127.0.0.1"], "boards": ["design"]})
            for task in (TASK_A, TASK_B):
                assert safety.page_data(call({"action": "open", "url": f"{base}/ok-timer?t={task}"}, task))["ok"]
            time.sleep(0.8)
            assert alive(seen, TASK_A) and alive(seen, TASK_B)
            yield agent, seen
        finally:
            design_browser_tool.cleanup_design_browser(TASK_A)
            design_browser_tool.cleanup_design_browser(TASK_B)
            agent.close()


@pytest.mark.integration
@pytest.mark.skipif(os.environ.get("HERMES_E2E_BROWSER", "").strip() != "1",
                    reason="real-browser E2E: set HERMES_E2E_BROWSER=1 to opt in")
@pytest.mark.skipif(not shutil.which("google-chrome") and not shutil.which("chromium"),
                    reason="Chrome/Chromium not installed")
class TestPublicCloseOnRealChromium:
    def test_close_ends_its_tasks_browser_and_leaves_the_other_task_running(self, two_tasks):
        agent, seen = two_tasks
        key_a, key_b = design_browser_tool.supervisor_key(TASK_A), design_browser_tool.supervisor_key(TASK_B)
        mine, theirs = design_browser_tool._SESSIONS[key_a], design_browser_tool._SESSIONS[key_b]
        agent.close()
        listed = contexts(key_b)
        assert SUPERVISOR_REGISTRY.get(key_a) is None and not mine.supervisor.snapshot().active
        assert key_a not in design_browser_tool._SESSIONS and key_a not in design_browser_tool._GUARDS
        assert mine.context_id not in listed
        assert theirs.context_id in listed and SUPERVISOR_REGISTRY.get(key_b) is theirs.supervisor
        before, after = background_requests(seen)
        assert after[TASK_A] == before[TASK_A]  # its page went with its context
        assert after[TASK_B] > before[TASK_B]
        assert safety.page_data(call({"action": "read_text"}, TASK_B))["ok"]

    def test_a_second_close_of_the_task_does_nothing(self, two_tasks):
        agent, seen = two_tasks
        key_a, key_b = design_browser_tool.supervisor_key(TASK_A), design_browser_tool.supervisor_key(TASK_B)
        mine, theirs = design_browser_tool._SESSIONS[key_a], design_browser_tool._SESSIONS[key_b]
        agent.close()
        listed = contexts(key_b)
        assert mine.context_id not in listed and theirs.context_id in listed
        agent.close()  # the same agent again
        new_agent(TASK_A).close()  # and a fresh agent for the same session
        assert sorted(contexts(key_b)) == sorted(listed)  # no context made or disposed
        assert SUPERVISOR_REGISTRY.get(key_a) is None
        assert key_a not in design_browser_tool._SESSIONS and key_a not in design_browser_tool._GUARDS
        assert design_browser_tool._SESSIONS[key_b] is theirs
        assert SUPERVISOR_REGISTRY.get(key_b) is theirs.supervisor and theirs.supervisor.snapshot().active
        before, after = background_requests(seen)
        assert after[TASK_A] == before[TASK_A] and after[TASK_B] > before[TASK_B]
        assert safety.page_data(call({"action": "read_text"}, TASK_B))["ok"]
