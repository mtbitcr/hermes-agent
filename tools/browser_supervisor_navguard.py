"""Navigation guard for the CDP supervisor (design-loop plan v2, B2).

Opt-in: only a ``CDPSupervisor`` built with ``navigation_guard=NavigationGuard(...)`` runs any of
this. On attach, before ``start()`` returns, it pauses every HTTP(S) request of the whole browser
(documents, each redirect hop, subresources, popups, frames and workers) at one browser-level
``Fetch`` interception, which applies to new targets from birth. A request goes on only if it is a
GET that :func:`tools.design_url_policy.design_url_refusal` allows; anything else, a policy error
included, is failed. Downloads and permissions are denied, file choosers are intercepted and never
answered, service-worker script fetches are failed, and every new target is held at start until
its share of the guard is installed, or closed if that fails.

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
from typing import Any, Awaitable, Callable, Dict, Optional

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
_PAGE_TYPES = frozenset({"page", "iframe"})


class NavigationGuard:
    """Allow-list guard for one supervisor. ``policy`` and ``builder_host`` are passed unchanged
    to :func:`design_url_refusal` (the ``design_browser.allowed_hosts`` list or the loaded config,
    and the builder's own preview host)."""

    def __init__(self, policy: Any, *, builder_host: Optional[str] = None) -> None:
        self.policy = policy
        self.builder_host = builder_host
        self._page_of: Dict[str, str] = {}  # attached session id -> target id of its page

    def refusal(self, request: Dict[str, Any]) -> Optional[str]:
        """Why a paused request must be failed, or None. Fails closed on a policy error."""
        method = request.get("method")
        if method != "GET":
            return f"{method} requests are not allowed"
        if any(str(name).lower() == "service-worker" for name in request.get("headers") or {}):
            return "service worker registration is blocked"
        try:
            return design_url_refusal(request.get("url"), self.policy, builder_host=self.builder_host)
        except Exception as exc:
            logger.warning("navigation guard: address policy failed, request refused: %s", type(exc).__name__)
            return "the address policy failed"

    async def install(self, cdp: CDPCall, page_session_id: str, page_target_id: str) -> None:
        """Guard the browser and the supervisor's first page. Raises if any step fails, so the
        supervisor does not start unguarded."""
        await cdp("Browser.setDownloadBehavior", {"behavior": "deny"})
        for name in DENIED_PERMISSIONS:
            await cdp("Browser.setPermission", {"permission": {"name": name}, "setting": "denied"})
        await cdp("Fetch.enable", {"patterns": FETCH_ALL, "handleAuthRequests": False})
        self._page_of = {page_session_id: page_target_id}  # sessions of a dropped connection are gone
        await self._lock_session(cdp, page_session_id, "page")
        # New pages (popups, tabs) are held at start until guard_target resumes them.
        await cdp("Target.setAutoAttach", _AUTO_ATTACH_HELD)

    def on_event(self, cdp: CDPCall, method: str, params: Dict[str, Any], session_id: Optional[str]) -> bool:
        """See a CDP event before the supervisor does. True when the guard consumed it. Work that
        awaits CDP replies runs in a task, because only the reader resolves those replies."""
        if method == "Fetch.requestPaused" and session_id is None:
            asyncio.create_task(self._answer(cdp, params))
            return True
        if method == "Target.attachedToTarget":
            asyncio.create_task(self.guard_target(cdp, params, session_id))
        elif method == "Network.webSocketCreated":
            asyncio.create_task(self._close_page(cdp, session_id, "it opened a WebSocket"))
        return False

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

    async def guard_target(self, cdp: CDPCall, params: Dict[str, Any], parent_session_id: Optional[str]) -> None:
        """Lock a newly attached target down, then let it run. If that fails it stays held and its
        page is closed."""
        sid = params.get("sessionId")
        info = params.get("targetInfo") or {}
        if not sid:
            return
        target_type = info.get("type")
        page = info.get("targetId") if target_type == "page" else self._page_of.get(parent_session_id or "")
        if page:
            self._page_of[sid] = page
        try:
            await self._lock_session(cdp, sid, target_type)
        except Exception as exc:
            await self._close_page(cdp, sid, f"its guard did not install ({exc})")
            return
        if params.get("waitingForDebugger"):
            await cdp("Runtime.runIfWaitingForDebugger", session_id=sid)

    async def _lock_session(self, cdp: CDPCall, sid: str, target_type: Optional[str]) -> None:
        """Network events (to see WebSockets) on every target, intercepted file choosers on pages
        and frames."""
        await cdp("Network.enable", session_id=sid)
        if target_type in _PAGE_TYPES:
            await cdp("Page.setInterceptFileChooserDialog", {"enabled": True}, session_id=sid)

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
