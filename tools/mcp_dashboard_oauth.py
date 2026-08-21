"""Dashboard-mediated callback bridge for MCP OAuth.

The MCP SDK remains responsible for discovery, DCR, PKCE, state validation and
token exchange. This module only moves the two human/browser callbacks from a
loopback listener into the already-authenticated dashboard session.
"""

from __future__ import annotations

import asyncio
import contextvars
import secrets
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator
from urllib.parse import parse_qs, urlparse


@dataclass
class DashboardOAuthFlow:
    flow_id: str
    server_name: str
    profile: str | None
    hermes_home: str
    redirect_uri: str
    reconnect_live: bool = False
    created_at: float = field(default_factory=time.time)
    status: str = "starting"
    authorization_url: str | None = None
    error: str | None = None
    tools: list[dict] = field(default_factory=list)
    expected_state: str | None = field(default=None, init=False)
    _callback: tuple[str, str | None] | None = field(default=None, init=False, repr=False)
    _callback_error: str | None = field(default=None, init=False, repr=False)
    _authorization_ready: threading.Event = field(default_factory=threading.Event, init=False, repr=False)
    _callback_ready: threading.Event = field(default_factory=threading.Event, init=False, repr=False)
    _worker_done: threading.Event = field(default_factory=threading.Event, init=False, repr=False)
    _cancelled: bool = field(default=False, init=False, repr=False)
    _discard_persisted_state: bool = field(default=False, init=False, repr=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)

    async def publish_authorization_url(self, url: str) -> None:
        state = parse_qs(urlparse(url).query).get("state", [None])[0]
        if not state:
            raise ValueError("OAuth authorization URL did not include state")
        with self._lock:
            if self.status in {"approved", "error"}:
                raise RuntimeError("OAuth flow already ended")
            self.expected_state = state
            self.authorization_url = url
            self.status = "authorization_required"
            self._authorization_ready.set()

    async def wait_for_authorization_url(self, timeout: float = 30.0) -> str:
        ready = await asyncio.to_thread(self._authorization_ready.wait, timeout)
        if not ready:
            raise TimeoutError("Timed out waiting for MCP authorization URL")
        if not self.authorization_url:
            raise RuntimeError(self.error or "MCP OAuth flow ended before authorization")
        return self.authorization_url

    def deliver_callback(
        self,
        *,
        code: str | None,
        state: str | None,
        error: str | None,
    ) -> None:
        with self._lock:
            if self._callback_ready.is_set():
                raise ValueError("OAuth callback already received")
            if (
                self.expected_state is None
                or state is None
                or not secrets.compare_digest(self.expected_state, state)
            ):
                raise ValueError("OAuth callback state mismatch")
            if error:
                self._callback_error = error
            elif code:
                self._callback = (code, state)
                self.status = "exchanging"
            else:
                self._callback_error = "OAuth callback did not include code or error"
            self._callback_ready.set()

    async def wait_for_callback(self, timeout: float = 300.0) -> tuple[str, str | None]:
        ready = await asyncio.to_thread(self._callback_ready.wait, timeout)
        if not ready:
            raise TimeoutError("Timed out waiting for MCP OAuth callback")
        if self._callback_error:
            raise RuntimeError(f"OAuth authorization failed: {self._callback_error}")
        if self._callback is None:
            raise RuntimeError("OAuth callback did not include an authorization code")
        return self._callback

    def mark_approved(self) -> None:
        with self._lock:
            if self.status == "error":
                raise RuntimeError("OAuth flow already ended")
            self.status = "approved"
            self.error = None

    def mark_error(self, error: str) -> None:
        with self._lock:
            if self.status == "approved" or self._cancelled:
                return
            self.status = "error"
            self.error = error
            self._authorization_ready.set()
            self._callback_ready.set()

    def cancel(self, error: str, *, discard_persisted_state: bool = False) -> None:
        """Stop the flow and serialize cancellation against OAuth state writes."""
        with self._lock:
            if self.status == "approved" and not discard_persisted_state:
                return
            self._cancelled = True
            self._discard_persisted_state = (
                self._discard_persisted_state or discard_persisted_state
            )
            self.status = "error"
            self.error = error
            self._authorization_ready.set()
            self._callback_ready.set()

    @contextmanager
    def persistence_guard(self, *, rollback: bool = False) -> Iterator[None]:
        """Prevent a cancelled flow from recreating OAuth or config state."""
        with self._lock:
            if self._cancelled and (
                not rollback or self._discard_persisted_state
            ):
                raise RuntimeError(self.error or "OAuth flow was cancelled")
            yield

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "flow_id": self.flow_id,
                "server_name": self.server_name,
                "status": self.status,
                "authorization_url": self.authorization_url,
                "error": self.error,
            }

    def mark_worker_done(self) -> None:
        self._worker_done.set()

    @property
    def worker_done(self) -> bool:
        return self._worker_done.is_set()


_current_dashboard_flow: contextvars.ContextVar[DashboardOAuthFlow | None] = (
    contextvars.ContextVar("mcp_dashboard_oauth_flow", default=None)
)


@contextmanager
def dashboard_oauth_flow(flow: DashboardOAuthFlow) -> Iterator[None]:
    token = _current_dashboard_flow.set(flow)
    try:
        yield
    finally:
        _current_dashboard_flow.reset(token)


def get_dashboard_oauth_flow() -> DashboardOAuthFlow | None:
    return _current_dashboard_flow.get()
