"""Navigation guard for the CDP supervisor (design-loop plan v2, B2).

Opt-in: only a ``CDPSupervisor`` built with ``navigation_guard=NavigationGuard(...)`` runs any of
this. It is active only in a browser that cannot open a new window: Chromium started with
``--block-new-web-contents`` (every ``window.open`` returns null, and no link, form or frame opens
a window) and ``--enable-automation`` (so that CDP hands out the command line). On attach it first
reads the browser's command line with ``Browser.getBrowserCommandLine``; if the read fails or the
first switch is not in it, it installs nothing and raises, so the supervisor neither starts nor
reports active and the design browser tool refuses, as with no browser.

Then, before ``start()`` returns, it pauses every HTTP(S) request of the whole browser
(documents, each redirect hop, subresources, frames and workers) at one browser-level ``Fetch``
interception, which applies to new targets from birth. A request goes on only if it is a GET that
:func:`tools.design_url_policy.design_url_refusal` allows; anything else, a policy error included,
is failed. Downloads and permissions are denied in every browser context (the default one, those
that exist at install, and the context of each new target before it runs), file choosers are
intercepted and never answered, service-worker script fetches are failed, and every new target is
held at start: a new window is closed there (below), and any other target is held until its share
of the guard is installed, or closed if that fails.

Documents Fetch never pauses (``blob:``, ``data:`` and other non-https documents) are checked by
the same policy: every page and frame gets a script, in its own world, that runs before anything
else in each new document. Unless the address is ``https://``, exactly ``about:blank`` or the error
page, it stops at a ``debugger;`` statement; the guard reads ``location.href`` there, and lets the
document go on only if the policy allows it. A refused document, or one the guard cannot check, is
never resumed and its page is closed. A document whose stop the guard never sees stops itself.

A new window that the page or any of its frames opens all the same (``window.open``, a link or
form with a target, a sandboxed frame's popup, a ``noopener`` one) is closed while it is held,
whatever its first address, as a second line of defence. Nothing is asked of its renderer first
(one held from birth has none yet to answer with), and the policy is not asked about it. It is then
let go as well, because being held also holds the window its opener is still inside
``window.open`` for. The close cannot stop a document the opener writes into the window's first
blank document at once; that is why the guard needs a browser that opens no window.

CDP Fetch never pauses a WebSocket handshake (checked against Chromium 153 by
``tests/tools/test_browser_navguard.py``), and neither ``Network.setBlockedURLs`` nor an injected
CSP header stopped one there. So a socket cannot be failed like a request: the page that opens
one is closed as soon as ``Network.webSocketCreated`` reports it, by which time the handshake may
already be on its way.

One guard serves one supervisor: it remembers which page each attached session belongs to.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable, Dict, Optional, Set

from tools.design_url_policy import design_url_refusal

# Logger-name parity with the supervisor (records must look unchanged).
logger = logging.getLogger("tools.browser_supervisor")

CDPCall = Callable[..., Awaitable[Dict[str, Any]]]

FETCH_ALL = [{"urlPattern": "*", "requestStage": "Request"}]
# Permission names Chromium accepts in Browser.setPermission; each is denied for every origin.
DENIED_PERMISSIONS = (
    "clipboard-read", "clipboard-write", "geolocation", "notifications", "camera", "microphone",
    "midi", "background-sync", "persistent-storage", "screen-wake-lock", "idle-detection",
)
_AUTO_ATTACH_HELD = {"autoAttach": True, "waitForDebuggerOnStart": True, "flatten": True}
# The switch without which the guard does not become active (owner rule 2).
NO_NEW_WINDOWS = "--block-new-web-contents"
_PAGE_TYPES = frozenset({"page", "iframe"})
_BROWSER_WORKERS = frozenset({"service_worker", "shared_worker"})  # browser-level auto-attach reaches these

# The document check. Its pause is told from others by the function's name; ``allowed`` is set
# from the paused frame when the policy allows the document.
GATE_WORLD = "hermes-navigation-guard"
GATE_FUNCTION = "hermesNavigationGate"
GATE_SOURCE = """(function hermesNavigationGate() {
  const url = String(location.href);
  if (url.startsWith("https://") || url === "about:blank" || url === "chrome-error://chromewebdata/") return;
  const hermesNavigationGate = { allowed: false };
  debugger;
  if (hermesNavigationGate.allowed) return;
  window.stop();
})();"""


class NavigationGuard:
    """Allow-list guard for one supervisor. ``policy`` and ``builder_host`` are passed unchanged
    to :func:`design_url_refusal` (the ``design_browser.allowed_hosts`` list or the loaded config,
    and the builder's own preview host)."""

    def __init__(self, policy: Any, *, builder_host: Optional[str] = None) -> None:
        self.policy = policy
        self.builder_host = builder_host
        self._page_of: Dict[str, str] = {}  # attached session id -> target id of its page
        self._held: Set[str] = set()  # targets held at start; their first session guards them
        self._own: Set[str] = set()  # sessions the supervisor never set up; their events stay here
        self._default_context: Optional[str] = None
        self._contexts: Dict[str, Awaitable[None]] = {}  # browser context id -> its denials

    def refusal(self, request: Dict[str, Any]) -> Optional[str]:
        """Why a paused request must be failed, or None. Fails closed on a policy error."""
        method = request.get("method")
        if method != "GET":
            return f"{method} requests are not allowed"
        if any(str(name).lower() == "service-worker" for name in request.get("headers") or {}):
            return "service worker registration is blocked"
        return self._url_refusal(request.get("url"))

    def _url_refusal(self, url: Any) -> Optional[str]:
        try:
            return design_url_refusal(url, self.policy, builder_host=self.builder_host)
        except Exception as exc:
            logger.warning("navigation guard: address policy failed, request refused: %s", type(exc).__name__)
            return "the address policy failed"

    async def install(self, cdp: CDPCall, page_session_id: str, page_target_id: str) -> None:
        """Guard the browser and the supervisor's first page. Raises if the browser can open new
        windows (or its command line cannot be read), or if any step fails, so the supervisor does
        not start unguarded."""
        switches = ((await cdp("Browser.getBrowserCommandLine")).get("result") or {}).get("arguments")
        if not isinstance(switches, list) or NO_NEW_WINDOWS not in switches:
            raise RuntimeError(f"the browser was not started with {NO_NEW_WINDOWS}")
        await self._deny(cdp)
        contexts = (await cdp("Target.getBrowserContexts")).get("result") or {}
        self._default_context = contexts.get("defaultBrowserContextId")
        self._contexts = {}
        for context_id in contexts.get("browserContextIds") or []:
            await self._deny_context(cdp, context_id)
        await cdp("Fetch.enable", {"patterns": FETCH_ALL, "handleAuthRequests": False})
        # Sessions of a dropped connection are gone.
        self._page_of = {page_session_id: page_target_id}
        self._held = set()
        self._own = set()
        await self._lock_session(cdp, page_session_id, "page", gate=True)
        # New pages (popups, tabs) are held at start until guard_target closes or resumes them.
        await cdp("Target.setAutoAttach", _AUTO_ATTACH_HELD)

    def on_event(self, cdp: CDPCall, method: str, params: Dict[str, Any], session_id: Optional[str]) -> bool:
        """See a CDP event before the supervisor does. True when the guard consumed it. Work that
        awaits CDP replies runs in a task, because only the reader resolves those replies."""
        if method == "Fetch.requestPaused" and session_id is None:
            asyncio.create_task(self._answer(cdp, params))
            return True
        if method == "Debugger.paused":
            asyncio.create_task(self._check_document(cdp, params, session_id))
        elif method == "Target.attachedToTarget":
            asyncio.create_task(self.guard_target(cdp, params, session_id))
        elif method == "Network.webSocketCreated":
            asyncio.create_task(self._close_page(cdp, session_id, "it opened a WebSocket"))
        # The supervisor never enables Debugger, nor anything on the sessions only the guard set up.
        return method.startswith("Debugger.") or session_id in self._own

    async def _answer(self, cdp: CDPCall, params: Dict[str, Any]) -> None:
        request_id = params.get("requestId")
        reason = self.refusal(params.get("request") or {})
        try:
            if reason is None:
                await cdp("Fetch.continueRequest", {"requestId": request_id})
            else:
                logger.info("navigation guard: refused a %s request: %s", params.get("resourceType"), reason)
                await cdp("Fetch.failRequest", {"requestId": request_id, "errorReason": "BlockedByClient"})
        except Exception as exc:
            logger.debug("navigation guard: answer for %s failed: %s", request_id, exc)

    async def _check_document(self, cdp: CDPCall, params: Dict[str, Any], sid: Optional[str]) -> None:
        """Answer a pause: the document check's goes on only if the policy allows the paused
        document; any other pause is resumed at once."""
        frame = (params.get("callFrames") or [{}])[0]
        if frame.get("functionName") != GATE_FUNCTION:
            try:
                await cdp("Debugger.resume", session_id=sid)
            except Exception as exc:
                logger.debug("navigation guard: resuming a pause failed: %s", exc)
            return
        frame_id = frame.get("callFrameId")
        try:
            href = await cdp("Debugger.evaluateOnCallFrame",
                             {"callFrameId": frame_id, "expression": "location.href", "returnByValue": True},
                             session_id=sid)
            reason = self._url_refusal(((href.get("result") or {}).get("result") or {}).get("value"))
            if reason is None:
                await cdp("Debugger.evaluateOnCallFrame",
                          {"callFrameId": frame_id, "expression": f"{GATE_FUNCTION}.allowed = true"}, session_id=sid)
                await cdp("Debugger.resume", session_id=sid)
                return
        except Exception as exc:
            reason = f"it could not be checked ({type(exc).__name__})"
        await self._close_page(cdp, sid, f"its document was refused: {reason}")

    async def guard_target(self, cdp: CDPCall, params: Dict[str, Any], parent_session_id: Optional[str]) -> None:
        """Close a new window while it is held; lock any other newly attached target down, then
        let it run. If that fails it stays held and its page is closed."""
        sid = params.get("sessionId")
        info = params.get("targetInfo") or {}
        if not sid:
            return
        target_type = info.get("type")
        target_id = info.get("targetId")
        page = target_id if target_type == "page" else self._page_of.get(parent_session_id or "")
        if page:
            self._page_of[sid] = page
        # Held targets are new. Pages that already ran (the supervisor's own among them, attached
        # again here) keep what they had; install() checks the documents of the supervisor's page.
        held = bool(params.get("waitingForDebugger"))
        if parent_session_id in self._own or (parent_session_id is None and held):
            self._own.add(sid)  # a new page (popup, tab) or a target inside one
        if held and parent_session_id is not None and target_type in _BROWSER_WORKERS:
            # Also attached (and guarded) through the browser, whose session gets no answer while
            # this one still holds the worker.
            try:
                await cdp("Runtime.runIfWaitingForDebugger", session_id=sid)
            except Exception as exc:  # a refused registration's worker is often gone already
                logger.debug("navigation guard: letting a worker go failed: %s", exc)
            return
        if held and target_type == "page" and (info.get("openerId") or info.get("openerFrameId")):
            # A new window the page or one of its frames opened, which --block-new-web-contents
            # should never let happen: closed while it is held, whatever its first address, as a
            # second line of defence. Nothing is asked of its renderer first (a page held from
            # birth has none yet, and would answer none of it), and the policy is not asked about it.
            # Closed, then let go, both sent before either answers. Being held holds the window its
            # opener is still inside window.open for, and only letting the page go hands that back
            # - but by the time the close answers the session is gone and the command would be
            # dropped. The close cannot stop what the opener writes into the window meanwhile.
            closing = asyncio.ensure_future(self._close_page(cdp, sid, "it is a new window a page opened"))
            letting_go = asyncio.ensure_future(cdp("Runtime.runIfWaitingForDebugger", session_id=sid))
            await closing
            try:
                await letting_go
            except Exception as exc:  # the session goes with the page it belonged to
                logger.debug("navigation guard: letting a closed page go: %s", exc)
            return
        repeat = target_id in self._held
        if held:
            self._held.add(target_id)
        try:
            await self._deny_context(cdp, info.get("browserContextId"))
            # One session per target checks its documents, or each pause would come twice.
            await self._lock_session(cdp, sid, target_type, gate=held and not repeat)
        except Exception as exc:
            await self._close_page(cdp, sid, f"its guard did not install ({exc})")
            return
        if held:
            await cdp("Runtime.runIfWaitingForDebugger", session_id=sid)

    async def _lock_session(self, cdp: CDPCall, sid: str, target_type: Optional[str], *, gate: bool) -> None:
        """Network events (to see WebSockets) on every target; on pages and frames, intercepted
        file choosers and (once per target) the document check, with their own new targets held."""
        await cdp("Network.enable", session_id=sid)
        if target_type not in _PAGE_TYPES:
            return
        await cdp("Page.setInterceptFileChooserDialog", {"enabled": True}, session_id=sid)
        if gate:
            for domain in ("Page", "Runtime", "Debugger"):
                await cdp(f"{domain}.enable", session_id=sid)
            await cdp("Page.addScriptToEvaluateOnNewDocument", {"source": GATE_SOURCE, "worldName": GATE_WORLD},
                      session_id=sid)
            await cdp("Target.setAutoAttach", _AUTO_ATTACH_HELD, session_id=sid)

    async def _deny(self, cdp: CDPCall, context_id: Optional[str] = None) -> None:
        """Downloads and every permission denied, in one browser context or the default one."""
        scope = {"browserContextId": context_id} if context_id else {}
        await cdp("Browser.setDownloadBehavior", {"behavior": "deny", **scope})
        for name in DENIED_PERMISSIONS:
            await cdp("Browser.setPermission", {"permission": {"name": name}, "setting": "denied", **scope})

    async def _deny_context(self, cdp: CDPCall, context_id: Optional[str]) -> None:
        """Deny in ``context_id`` once; raises if that fails, and the next target there tries again."""
        if not context_id or context_id == self._default_context:  # denied at install
            return
        pending = self._contexts.get(context_id)
        if pending is None:
            pending = self._contexts[context_id] = asyncio.ensure_future(self._deny(cdp, context_id))
        try:
            await pending
        except Exception:
            if self._contexts.get(context_id) is pending:
                del self._contexts[context_id]
            raise

    async def _close_page(self, cdp: CDPCall, session_id: Optional[str], why: str) -> None:
        target_id = self._page_of.get(session_id or "")
        logger.warning("navigation guard: closing a page because %s", why)
        if target_id is None:
            return
        try:
            await cdp("Target.closeTarget", {"targetId": target_id})
        except Exception as exc:
            logger.debug("navigation guard: closing %s failed: %s", target_id, exc)


__all__ = ["DENIED_PERMISSIONS", "NavigationGuard"]
