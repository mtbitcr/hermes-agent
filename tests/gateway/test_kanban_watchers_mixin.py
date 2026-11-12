"""Tests for the extracted GatewayKanbanWatchersMixin (god-file Phase 3).

The kanban watcher loops were lifted out of gateway/run.py into a mixin that
GatewayRunner inherits. These tests confirm the mixin exposes the methods and
that GatewayRunner picks them up via the MRO (behavior-neutral relocation).
"""

from __future__ import annotations

import asyncio
import inspect
from types import SimpleNamespace

from gateway.kanban_watchers import GatewayKanbanWatchersMixin

KANBAN_METHODS = [
    "_kanban_notifier_watcher",
    "_kanban_dispatcher_watcher",
    "_kanban_advance",
    "_kanban_unsub",
    "_kanban_rewind",
    "_deliver_kanban_artifacts",
]


def test_mixin_defines_kanban_methods():
    for m in KANBAN_METHODS:
        assert hasattr(GatewayKanbanWatchersMixin, m), f"mixin missing {m}"


def test_the_gateway_runner_has_the_daily_digest_watcher():
    from gateway.run import GatewayRunner

    assert inspect.iscoroutinefunction(GatewayRunner._owner_review_digest_watcher)


def test_gateway_start_up_starts_the_digest_watcher_with_the_other_pre_reconnect_watchers():
    from gateway.run import GatewayRunner

    runner = GatewayRunner.__new__(GatewayRunner)
    started = []
    runner._spawn_supervised = lambda factory, name, **_kwargs: started.append((name, factory))
    runner._failed_platforms = {}
    runner._spawn_reconnect_watcher = lambda: started.append(("reconnect_watcher", None))
    runner._scale_to_zero_should_arm = lambda: False
    runner._log_scale_to_zero_not_armed_reason = lambda: None
    runner._start_spawn_background_watchers()
    names = [name for name, _factory in started]
    assert dict(started)["owner_review_digest_watcher"] == runner._owner_review_digest_watcher
    assert names.index("owner_heads_up_watcher") < names.index("owner_review_digest_watcher")
    assert names.index("owner_review_digest_watcher") < names.index("reconnect_watcher")


def test_the_digest_watcher_ticks_survives_a_failing_tick_and_stops_with_the_runner(monkeypatch):
    from gateway import owner_review_digest
    from gateway.run_watchers import GatewaySessionWatchersMixin

    runner = SimpleNamespace(_running=True)
    calls = []

    async def tick(passed):
        calls.append(passed)
        if len(calls) == 1:
            raise RuntimeError("synthetic tick failure")
        if len(calls) == 3:
            runner._running = False
        return 0

    monkeypatch.setattr(owner_review_digest, "tick", tick)
    watcher = GatewaySessionWatchersMixin._owner_review_digest_watcher(runner, interval=0)
    asyncio.run(asyncio.wait_for(watcher, 10))
    assert calls == [runner, runner, runner]
