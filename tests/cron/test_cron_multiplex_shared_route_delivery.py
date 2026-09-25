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

        # unmatched chat, disabled route, route for another profile → refused:
        # the primary bot is NEVER used and the satellite's standalone path is
        # never tried.
        for chat in ("424242", "999", "777"):
            primary.sent.clear()
            error, standalone = _run(_job(chat), shared)
            assert error is not None
            assert primary.sent == []
            assert standalone == []
    finally:
        reset_hermes_home_override(token)


def test_chat_only_route_lends_the_primary_adapter_for_the_bare_chat_only(tmp_path, monkeypatch):
    """A route that names only a chat lends the main bot for that bare chat, never for a target that
    adds a thread: Discord sends to the thread part as a channel, so a job could otherwise post
    through the main bot into any channel it can see."""
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
        error, standalone = _run(_job("1543065293755256852"), shared)
        assert error is None, error
        assert primary.sent == ["1543065293755256852"] and standalone == []

        primary.sent.clear()
        error, standalone = _run(_job("1543065293755256852:9"), shared)
        assert error is not None
        assert primary.sent == [] and standalone == []
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


def _weekly_report(job, **kwargs):
    return True, "out", "Weekly report", None


def _tick_planning_briefs(
    tmp_path, monkeypatch, delivery_only, *, adapters, planning_adapters, run_job=_weekly_report,
    routes=None, jobs=None, launch=None, jobs_by_profile=None,
):
    """Fire planning's owner, stray and ops telegram briefs (or the given ``jobs`` under the given
    ``routes``) through the regular multiplex ticker (``_start_multiplex`` → ``cron.scheduler.tick``).
    ``adapters`` is the launch profile's live map, ``planning_adapters`` planning's own. With
    ``launch`` the gateway runs as ``hermes -p <launch> gateway``: the process home and env are that
    profile's, it owns ``adapters``, and the root profile is served as a secondary; then
    ``jobs_by_profile`` maps a profile name (``None`` for the root) to its jobs. Returns
    ``(delivery_errors, standalone)``; ``standalone`` holds ``(chat_id, token)`` for every
    standalone ``_send_to_platform`` call."""
    from cron.scheduler_provider import InProcessCronScheduler
    from hermes_constants import get_hermes_home

    root = tmp_path / "root"
    planning_home = root / "profiles" / "planning"
    (planning_home / "cron").mkdir(parents=True)
    (root / "cron").mkdir()
    homes = {None: root, "planning": planning_home}
    served = [(None, root), ("planning", planning_home)]
    profile_adapters = {"planning": planning_adapters}
    if launch is not None:
        homes[launch] = root / "profiles" / launch
        (homes[launch] / "cron").mkdir(parents=True)
        monkeypatch.setenv("HERMES_HOME", str(homes[launch]))
        served, profile_adapters = [(launch, homes[launch]), (None, root)], {}
    if routes is None:
        routes = [
            {"name": "owner-reports", "platform": "telegram", "chat_id": OWNER_CHAT,
             "profile": "planning", "delivery_only": delivery_only},
            {"name": "ops", "platform": "telegram", "chat_id": OPS_CHAT, "profile": "ops"},
        ]
    (root / "config.yaml").write_text(yaml.safe_dump({"gateway": {
        "multiplex_profiles": True,
        "profile_routes": routes,
    }}), encoding="utf-8")
    monkeypatch.setattr("hermes_constants.get_default_hermes_root", lambda: root)

    if jobs is None:
        jobs = [
            {"id": "owner-brief", "name": "owner brief", "deliver": f"telegram:{OWNER_CHAT}"},
            {"id": "stray-brief", "name": "stray brief", "deliver": f"telegram:{STRAY_CHAT}"},
            {"id": "ops-brief", "name": "ops brief", "deliver": f"telegram:{OPS_CHAT}"},
        ]
    jobs_by_home = {
        homes[name].resolve(): list(profile_jobs)
        for name, profile_jobs in (jobs_by_profile or {"planning": jobs}).items()
    }
    job_ids = {job["id"] for profile_jobs in jobs_by_home.values() for job in profile_jobs}
    handed_out = set()

    def planning_due_jobs():
        # Each profile's jobs live in its own store only; hand them out on its first tick.
        home = Path(get_hermes_home()).resolve()
        if home in handed_out or home not in jobs_by_home:
            return []
        handed_out.add(home)
        return [dict(job) for job in jobs_by_home[home]]

    delivery_errors = {}

    def record_run(job_id, success, error=None, **kwargs):
        delivery_errors[job_id] = kwargs.get("delivery_error")
        return True

    standalone = []

    async def standalone_send(platform, pconfig, chat_id, *args, **kwargs):
        # The token is whatever the unscoped gateway config handed the standalone path.
        standalone.append((chat_id, getattr(pconfig, "token", None)))
        return {"success": True, "message_id": "standalone-1"}

    loop = asyncio.new_event_loop()  # the gateway's running loop
    loop_thread = threading.Thread(target=loop.run_forever, daemon=True)
    loop_thread.start()
    stop = threading.Event()
    try:
        with patch("cron.scheduler.get_due_jobs", side_effect=planning_due_jobs), \
             patch("cron.scheduler.advance_next_runs"), \
             patch("cron.scheduler.claim_job_for_fire", return_value=True), \
             patch("cron.scheduler.run_job", side_effect=run_job), \
             patch("cron.scheduler.save_job_output", return_value=str(tmp_path / "out.md")), \
             patch("cron.scheduler.mark_job_run", side_effect=record_run), \
             patch("cron.jobs.record_ticker_heartbeat", lambda **kw: None), \
             patch("tools.send_message_tool._send_to_platform", standalone_send):
            ticker = threading.Thread(target=InProcessCronScheduler().start, args=(stop,), kwargs={
                "interval": 0, "loop": loop,
                "profile_homes": served,
                "adapters": adapters,
                "profile_adapters": profile_adapters,
                "default_profile": launch,
            }, daemon=True)
            ticker.start()
            deadline = time.monotonic() + 30
            while len(delivery_errors) < len(job_ids) and time.monotonic() < deadline:
                time.sleep(0.01)
            stop.set()
            ticker.join(timeout=10)
    finally:
        loop.call_soon_threadsafe(loop.stop)
        loop_thread.join(timeout=5)
        loop.close()

    assert set(delivery_errors) == job_ids
    return delivery_errors, standalone


@pytest.mark.parametrize("delivery_only", [True, False])
def test_multiplex_ticker_delivers_satellite_report_only_to_its_routed_chat(
    tmp_path, monkeypatch, delivery_only,
):
    """Through the regular multiplex ticker (``_start_multiplex`` → ``cron.scheduler.tick``), a
    credential-less satellite's scheduled report reaches the owner's chat through the PRIMARY bot,
    because an enabled primary route names that exact chat for this satellite (delivery-only or
    ordinary alike). Any other target — unrouted, or routed to another profile — is refused."""
    primary = _primary_adapter()
    delivery_errors, standalone = _tick_planning_briefs(
        tmp_path, monkeypatch, delivery_only,
        adapters={Platform.TELEGRAM: primary},
        planning_adapters={},  # planning holds no bot of its own
    )
    assert delivery_errors["owner-brief"] is None
    assert primary.sent == [OWNER_CHAT]  # the primary bot reached exactly the routed chat
    assert delivery_errors["stray-brief"] and delivery_errors["ops-brief"]  # refused
    assert standalone == []


@pytest.mark.parametrize("delivery_only", [True, False])
def test_multiplex_ticker_refuses_other_targets_when_process_env_holds_main_bot_token(
    tmp_path, monkeypatch, delivery_only,
):
    """The normal multiplex deployment: the gateway process env holds the main bot's token (the
    launch profile's .env), so the satellite's unscoped gateway config enables telegram although
    planning has no telegram section. The report still reaches only the routed chat, through the
    PRIMARY adapter; the stray and ops briefs are refused and nothing is sent standalone."""
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "placeholder-bot-token")
    test_multiplex_ticker_delivers_satellite_report_only_to_its_routed_chat(
        tmp_path, monkeypatch, delivery_only,
    )


@pytest.mark.parametrize("delivery_only", [True, False])
@pytest.mark.parametrize("shape", [
    pytest.param("own_bot_on_discord", id="B-own-discord-bot"),
    pytest.param("empty_main_map", id="A-empty-main-map"),
])
def test_multiplex_ticker_refuses_unrouted_targets_whatever_the_adapter_map(
    tmp_path, monkeypatch, shape, delivery_only,
):
    """The refusal of a target no route names for this profile must not depend on the ticker
    handing the profile a SharedRouteAdapters map. Two ordinary shapes hand it a plain map instead:
    (B) planning runs its own bot on another platform, so its own map is not empty; (A) the main
    adapter map is empty, e.g. while its Telegram adapter is popped for a reconnect. The process env
    holds the main bot's token (a placeholder), and the real preflight runs inside run_job under
    planning's secret scope with multiplex active; its per-platform route exception lets all three
    briefs run. No brief may reach a chat no route names for planning: not through the main
    adapter, not through planning's own adapter, not through the standalone send."""
    from agent import secret_scope
    from cron import scheduler as live_scheduler

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "placeholder-bot-token")
    primary, own = _primary_adapter(), _primary_adapter()
    if shape == "own_bot_on_discord":
        adapters, planning_adapters = {Platform.TELEGRAM: primary}, {Platform.DISCORD: own}
    else:
        adapters, planning_adapters = {}, {}

    preflight = {}

    def run_job_behind_real_preflight(job, **kwargs):
        # run_one_job has installed planning's secret scope around this call.
        scoped_token = secret_scope.get_secret("TELEGRAM_BOT_TOKEN")
        reason = live_scheduler._preflight_check_delivery(job)
        preflight[job["id"]] = (scoped_token, reason)
        if reason:  # what run_job returns when its preflight blocks
            return False, reason, "", f"{live_scheduler.BLOCKED_CONFIG_MARKER} {reason}"
        return _weekly_report(job)

    was_multiplex = secret_scope.is_multiplex_active()
    secret_scope.set_multiplex_active(True)
    try:
        delivery_errors, standalone = _tick_planning_briefs(
            tmp_path, monkeypatch, delivery_only,
            adapters=adapters, planning_adapters=planning_adapters,
            run_job=run_job_behind_real_preflight,
        )
    finally:
        secret_scope.set_multiplex_active(was_multiplex)

    # Planning's scope holds no bot token, and the route exception let every brief run.
    assert preflight == {job_id: (None, None) for job_id in ("owner-brief", "stray-brief", "ops-brief")}
    sends = (
        [("main adapter", chat) for chat in primary.sent]
        + [("planning's own adapter", chat) for chat in own.sent]
        + [(f"standalone send, token={token}", chat) for chat, token in standalone]
    )
    assert [send for send in sends if send[1] in (STRAY_CHAT, OPS_CHAT)] == []
    for job_id in ("stray-brief", "ops-brief"):
        assert "not named by an enabled route for this profile" in delivery_errors[job_id]
    # The routed brief still goes on: once, and only to the routed chat.
    assert [chat for _, chat in sends] == [OWNER_CHAT]
    assert delivery_errors["owner-brief"] is None


@pytest.mark.parametrize("shape", [
    pytest.param("no_route", id="no-route"),
    pytest.param("routed_on_discord_only", id="routed-on-another-platform"),
])
def test_multiplex_ticker_refuses_a_satellite_target_on_a_platform_no_route_names(
    tmp_path, monkeypatch, shape,
):
    """A satellite under multiplex reaches a chat only through its own live adapter or a route that
    names that exact chat. On a platform no route names for it, an explicit target and the home
    channel that "all" resolves from the process env are both refused: the standalone send would
    carry the main bot's token, which the unscoped delivery config reads from the process env."""
    from agent import secret_scope

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "placeholder-bot-token")
    monkeypatch.setenv("TELEGRAM_HOME_CHANNEL", "111")
    primary = _primary_adapter()
    routes = [{"name": "ops", "platform": "telegram", "chat_id": OPS_CHAT, "profile": "ops"}]
    planning_adapters = {}
    if shape == "routed_on_discord_only":
        routes.append({"name": "planning-discord", "platform": "discord", "chat_id": "424242",
                       "profile": "planning"})
        planning_adapters = {Platform.DISCORD: _primary_adapter()}
    jobs = [
        {"id": "explicit-brief", "name": "explicit brief", "deliver": "telegram:222"},
        {"id": "all-brief", "name": "all brief", "deliver": "all"},
    ]

    was_multiplex = secret_scope.is_multiplex_active()
    secret_scope.set_multiplex_active(True)
    try:
        delivery_errors, standalone = _tick_planning_briefs(
            tmp_path, monkeypatch, False,
            adapters={Platform.TELEGRAM: primary}, planning_adapters=planning_adapters,
            routes=routes, jobs=jobs,
        )
    finally:
        secret_scope.set_multiplex_active(was_multiplex)

    assert primary.sent == []
    assert standalone == []
    assert "not named by an enabled route for this profile" in delivery_errors["explicit-brief"]
    # "all" resolved the home channel and was refused, not left unresolved.
    assert "not named by an enabled route for this profile" in delivery_errors["all-brief"]


def _tick_under_named_launch(tmp_path, monkeypatch, jobs_by_profile):
    """Run the given profiles' jobs under a multiplex gateway launched as ``hermes -p coder
    gateway`` whose live map holds only a Telegram adapter; no route names any profile."""
    from agent import secret_scope

    primary = _primary_adapter()
    was_multiplex = secret_scope.is_multiplex_active()
    secret_scope.set_multiplex_active(True)
    try:
        delivery_errors, standalone = _tick_planning_briefs(
            tmp_path, monkeypatch, False,
            adapters={Platform.TELEGRAM: primary}, planning_adapters={},
            routes=[], launch="coder", jobs_by_profile=jobs_by_profile,
        )
    finally:
        secret_scope.set_multiplex_active(was_multiplex)
    return delivery_errors, standalone, primary


def test_named_launch_profile_keeps_its_own_delivery_on_a_platform_without_a_live_adapter(
    tmp_path, monkeypatch,
):
    """The launch profile of a multiplex gateway is the primary, not a satellite, even when it is
    not the root: its process env and live adapters are its own. Its job to a platform missing
    from its live map keeps its own path, the standalone send with its own settings."""
    monkeypatch.setenv("DISCORD_BOT_TOKEN", "placeholder-coder-discord-token")
    delivery_errors, standalone, primary = _tick_under_named_launch(tmp_path, monkeypatch, {
        "coder": [{"id": "coder-brief", "name": "coder brief", "deliver": "discord:424242"}],
    })
    assert delivery_errors["coder-brief"] is None
    assert standalone == [("424242", "placeholder-coder-discord-token")]
    assert primary.sent == []


def test_named_launch_serves_the_root_profile_as_a_satellite(tmp_path, monkeypatch):
    """Under a gateway launched as ``hermes -p coder gateway`` the root profile is a secondary and
    the process env holds coder's bot token. With no route naming the root profile, its explicit
    target and its "all" home target are refused: nothing goes through coder's adapter or the
    standalone send."""
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "placeholder-bot-token")
    monkeypatch.setenv("TELEGRAM_HOME_CHANNEL", "111")
    delivery_errors, standalone, primary = _tick_under_named_launch(tmp_path, monkeypatch, {
        None: [
            {"id": "root-explicit", "name": "root explicit", "deliver": "telegram:222"},
            {"id": "root-all", "name": "root all", "deliver": "all"},
        ],
    })
    assert primary.sent == []
    assert standalone == []
    for job_id in ("root-explicit", "root-all"):
        assert "not named by an enabled route for this profile" in delivery_errors[job_id]


def test_thread_only_route_lends_the_main_bot_for_no_chat():
    """Telegram topic ids and Slack thread timestamps are per chat, so a route that names only a
    thread would lend the main bot for any chat carrying that thread id. A cron target always
    carries a chat, so only a route that names the chat lends."""
    from gateway.profile_routing import parse_profile_routes

    shared = SharedRouteAdapters({Platform.TELEGRAM: _primary_adapter()}, parse_profile_routes([
        {"name": "topic", "platform": "telegram", "thread_id": "14", "profile": "planning"},
    ]))
    assert shared.get(Platform.TELEGRAM, {"chat_id": "-100999", "thread_id": "14"}) is None
    assert shared.get(Platform.TELEGRAM, {"chat_id": OWNER_CHAT, "thread_id": "14"}) is None


def test_chat_and_thread_route_lends_the_main_bot_for_exactly_that_topic():
    from gateway.profile_routing import parse_profile_routes

    primary = _primary_adapter()
    shared = SharedRouteAdapters({Platform.TELEGRAM: primary}, parse_profile_routes([
        {"name": "topic", "platform": "telegram", "chat_id": OPS_CHAT, "thread_id": "14",
         "profile": "planning"},
    ]))
    assert shared.get(Platform.TELEGRAM, {"chat_id": OPS_CHAT, "thread_id": "14"}) is primary
    for target in (
        {"chat_id": OPS_CHAT, "thread_id": "15"},
        {"chat_id": "-100999", "thread_id": "14"},
        {"chat_id": OPS_CHAT},
    ):
        assert shared.get(Platform.TELEGRAM, target) is None


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
