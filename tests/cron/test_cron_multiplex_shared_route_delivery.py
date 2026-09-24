"""Regression tests for #101113 — a credentialless satellite profile under
``gateway.profile_routes`` delivers cron output through the PRIMARY adapter
for exactly the targets the primary routes to it, and fails closed otherwise.

The multiplex ticker hands such a profile a ``SharedRouteAdapters`` view over
the primary adapter map; ``_deliver_result`` resolves a transport from it per
target using the same ``ProfileRoute.matches`` predicate as inbound routing.
"""
import asyncio
import threading
import time
from concurrent.futures import Future
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from cron.scheduler import _deliver_result
from cron.scheduler_preflight import SharedRouteAdapters, _primary_profile_routes_for_current_home
from gateway.config import Platform, PlatformConfig
from hermes_constants import reset_hermes_home_override, set_hermes_home_override

PRIMARY_YAML = {
    "gateway": {
        "multiplex_profiles": True,
        "profile_routes": [
            {"name": "fit", "platform": "discord", "chat_id": "1543065293755256852", "profile": "fitness"},
            {"name": "off", "platform": "discord", "chat_id": "999", "profile": "fitness", "enabled": False},
            {"name": "other", "platform": "discord", "chat_id": "777", "profile": "other"},
        ],
    }
}


def _job(chat_id: str) -> dict:
    return {"id": "a7ae1520356c", "name": "brief", "deliver": f"discord:{chat_id}"}


def _run(job, adapters):
    """Drive ``_deliver_result`` with a live loop and a real DeliveryRouter."""
    loop = MagicMock()
    loop.is_running.return_value = True

    def fake_run_coro(coro, _loop):
        future = Future()
        future.set_result(asyncio.run(coro))
        return future

    standalone = []

    async def _fake_send_to_platform(platform, pconfig, chat_id, text, **kwargs):
        standalone.append(chat_id)
        return {"success": False, "error": "DISCORD_BOT_TOKEN is not set"}

    config = MagicMock()
    config.platforms = {Platform.DISCORD: PlatformConfig(enabled=True)}
    config.get_home_channel = lambda p: None
    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}), \
         patch("tools.send_message_tool._send_to_platform", _fake_send_to_platform), \
         patch("asyncio.run_coroutine_threadsafe", side_effect=fake_run_coro):
        error = _deliver_result(job, "hello", adapters=adapters, loop=loop)
    return error, standalone


def _primary_adapter():
    adapter = MagicMock()
    adapter.sent = []

    async def send(chat_id, content, metadata=None):
        adapter.sent.append(chat_id)
        return {"success": True, "message_id": "m1"}

    adapter.send = send
    return adapter


def test_satellite_routes_exact_target_through_primary_adapter(tmp_path, monkeypatch):
    root = tmp_path / "root"
    fitness_home = root / "profiles" / "fitness"
    fitness_home.mkdir(parents=True)
    (root / "config.yaml").write_text(yaml.safe_dump(PRIMARY_YAML), encoding="utf-8")
    monkeypatch.setattr("hermes_constants.get_default_hermes_root", lambda: root)
    primary = _primary_adapter()

    token = set_hermes_home_override(str(fitness_home))
    try:
        shared = SharedRouteAdapters(
            {Platform.DISCORD: primary}, _primary_profile_routes_for_current_home()
        )
        # exact enabled route → primary adapter sends, no standalone attempt
        error, standalone = _run(_job("1543065293755256852"), shared)
        assert error is None, error
        assert primary.sent == ["1543065293755256852"]
        assert standalone == []

        # unmatched chat, disabled route, route for another profile → the
        # primary bot is NEVER used; delivery stays on the satellite's own
        # (credentialless) standalone path and reports its failure.
        for chat in ("424242", "999", "777"):
            primary.sent.clear()
            error, standalone = _run(_job(chat), shared)
            assert error is not None and "DISCORD_BOT_TOKEN" in error
            assert primary.sent == []
            assert standalone == [chat]
    finally:
        reset_hermes_home_override(token)


def test_shared_view_is_falsy_without_routes_or_primary_adapters():
    assert not SharedRouteAdapters({}, [])
    assert SharedRouteAdapters({Platform.DISCORD: object()}, []).get(Platform.DISCORD) is None


def test_guild_scoped_route_authorizes_cron_target_even_when_satellite_has_no_platform_block(
    tmp_path, monkeypatch,
):
    """The documented Discord route shape is ``guild_id + chat_id``. A cron target carries no guild
    anchor, so the route must be matched on its target-exact discriminators; and the satellite's
    missing/disabled ``platforms.discord`` block must not veto the PRIMARY's authorized transport
    (#89302 sibling) — before, both fell to standalone "DISCORD_BOT_TOKEN is not set"."""
    root = tmp_path / "root"
    sat_home = root / "profiles" / "fitness"
    sat_home.mkdir(parents=True)
    (root / "config.yaml").write_text(yaml.safe_dump({
        "gateway": {"multiplex_profiles": True, "profile_routes": [
            {"platform": "discord", "guild_id": "G1", "chat_id": "C1", "profile": "fitness"}]},
    }), encoding="utf-8")
    monkeypatch.setattr("hermes_constants.get_default_hermes_root", lambda: root)
    primary = _primary_adapter()

    token = set_hermes_home_override(str(sat_home))
    try:
        shared = SharedRouteAdapters({Platform.DISCORD: primary}, _primary_profile_routes_for_current_home())
        for satellite_platforms in ({}, {Platform.DISCORD: PlatformConfig(enabled=False)}):
            primary.sent.clear()
            config = MagicMock()
            config.platforms = satellite_platforms
            config.get_home_channel = lambda p: None
            with patch("gateway.config.load_gateway_config", return_value=config):
                error, standalone = _run(_job("C1"), shared)
            assert error is None, error
            assert primary.sent == ["C1"] and standalone == []
    finally:
        reset_hermes_home_override(token)


OWNER_CHAT = "640466638"
STRAY_CHAT = "555000111"
OPS_CHAT = "-1002223334445"


@pytest.mark.parametrize("delivery_only", [True, False])
def test_multiplex_ticker_delivers_satellite_report_only_to_its_routed_chat(
    tmp_path, monkeypatch, delivery_only,
):
    """Through the regular multiplex ticker (``_start_multiplex`` → ``cron.scheduler.tick``), a
    credential-less satellite's scheduled report reaches the owner's chat through the PRIMARY bot,
    because an enabled primary route names that exact chat for this satellite (delivery-only or
    ordinary alike). Any other target — unrouted, or routed to another profile — is refused."""
    from cron.scheduler_provider import InProcessCronScheduler
    from hermes_constants import get_hermes_home

    root = tmp_path / "root"
    planning_home = root / "profiles" / "planning"
    (planning_home / "cron").mkdir(parents=True)
    (root / "cron").mkdir()
    (root / "config.yaml").write_text(yaml.safe_dump({"gateway": {
        "multiplex_profiles": True,
        "profile_routes": [
            {"name": "owner-reports", "platform": "telegram", "chat_id": OWNER_CHAT,
             "profile": "planning", "delivery_only": delivery_only},
            {"name": "ops", "platform": "telegram", "chat_id": OPS_CHAT, "profile": "ops"},
        ],
    }}), encoding="utf-8")
    monkeypatch.setattr("hermes_constants.get_default_hermes_root", lambda: root)

    jobs = [
        {"id": "owner-brief", "name": "owner brief", "deliver": f"telegram:{OWNER_CHAT}"},
        {"id": "stray-brief", "name": "stray brief", "deliver": f"telegram:{STRAY_CHAT}"},
        {"id": "ops-brief", "name": "ops brief", "deliver": f"telegram:{OPS_CHAT}"},
    ]
    handed_out = threading.Event()

    def planning_due_jobs():
        # These jobs live in planning's store only; hand them out on its first tick.
        if handed_out.is_set() or Path(get_hermes_home()).resolve() != planning_home.resolve():
            return []
        handed_out.set()
        return [dict(job) for job in jobs]

    delivery_errors = {}

    def record_run(job_id, success, error=None, **kwargs):
        delivery_errors[job_id] = kwargs.get("delivery_error")
        return True

    standalone = []

    async def standalone_send(platform, pconfig, chat_id, *args, **kwargs):
        standalone.append(chat_id)
        return {"error": "planning holds no telegram credential"}

    primary = _primary_adapter()
    loop = asyncio.new_event_loop()  # the gateway's running loop
    loop_thread = threading.Thread(target=loop.run_forever, daemon=True)
    loop_thread.start()
    stop = threading.Event()
    try:
        with patch("cron.scheduler.get_due_jobs", side_effect=planning_due_jobs), \
             patch("cron.scheduler.advance_next_runs"), \
             patch("cron.scheduler.claim_job_for_fire", return_value=True), \
             patch("cron.scheduler.run_job", return_value=(True, "out", "Weekly report", None)), \
             patch("cron.scheduler.save_job_output", return_value=str(tmp_path / "out.md")), \
             patch("cron.scheduler.mark_job_run", side_effect=record_run), \
             patch("cron.jobs.record_ticker_heartbeat", lambda **kw: None), \
             patch("tools.send_message_tool._send_to_platform", standalone_send):
            ticker = threading.Thread(target=InProcessCronScheduler().start, args=(stop,), kwargs={
                "interval": 0, "loop": loop,
                "profile_homes": [(None, root), ("planning", planning_home)],
                "adapters": {Platform.TELEGRAM: primary},
                "profile_adapters": {"planning": {}},  # planning holds no bot of its own
                "default_profile": None,
            }, daemon=True)
            ticker.start()
            deadline = time.monotonic() + 30
            while len(delivery_errors) < len(jobs) and time.monotonic() < deadline:
                time.sleep(0.01)
            stop.set()
            ticker.join(timeout=10)
    finally:
        loop.call_soon_threadsafe(loop.stop)
        loop_thread.join(timeout=5)
        loop.close()

    assert set(delivery_errors) == {job["id"] for job in jobs}
    assert delivery_errors["owner-brief"] is None
    assert primary.sent == [OWNER_CHAT]  # the primary bot reached exactly the routed chat
    assert delivery_errors["stray-brief"] and delivery_errors["ops-brief"]  # refused
    assert standalone == []


def test_live_native_adapter_without_platform_block_is_not_treated_as_disabled():
    """#89302: a live native adapter handed in by the gateway is the authorization; an absent
    ``platforms.<p>`` block in the firing profile means "no config", not "disabled"."""
    from cron.scheduler_delivery import _resolve_target_transport

    config = MagicMock()
    config.platforms = {}
    adapter = object()
    resolved, err = _resolve_target_transport(
        {"id": "j"}, Platform.DISCORD, "discord", {"platform": "discord", "chat_id": "C1"},
        {Platform.DISCORD: adapter}, config)
    assert err is None and resolved[2] is adapter and resolved[1].enabled
    # an explicitly disabled block still vetoes
    config.platforms = {Platform.DISCORD: PlatformConfig(enabled=False)}
    resolved, err = _resolve_target_transport(
        {"id": "j"}, Platform.DISCORD, "discord", {"platform": "discord", "chat_id": "C1"},
        {Platform.DISCORD: adapter}, config)
    assert resolved is None and "not configured/enabled" in err
