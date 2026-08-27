"""HERMES_DISABLE_FALLBACKS must suppress EVERY auxiliary substitution path.

The kanban dispatcher sets this env var for a model-policy-locked worker: that
run is pinned to one exact owner-approved route, so no auxiliary path may
quietly resolve a different provider or model — not the configured per-task
chain, not the main agent's chain, not the built-in discovery chain reached
after a payment error, and not the main-agent-model safety net.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent import auxiliary_client as aux
from hermes_cli.fallback_config import FALLBACKS_DISABLED_ENV, get_fallback_chain

_REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def killed(monkeypatch):
    monkeypatch.setenv(FALLBACKS_DISABLED_ENV, "1")


@pytest.fixture
def allowed(monkeypatch):
    monkeypatch.delenv(FALLBACKS_DISABLED_ENV, raising=False)


def _sentinel_chain():
    """A provider chain whose every entry would succeed if it were consulted."""
    return [("openrouter", lambda: (object(), "some-model"))]


def test_configured_per_task_chain_is_suppressed(killed, monkeypatch):
    monkeypatch.setattr(
        aux, "_get_auxiliary_task_config",
        lambda task: {"fallback_chain": [{"provider": "openrouter", "model": "m"}]},
    )
    monkeypatch.setattr(
        aux, "_resolve_fallback_entry", lambda entry: (object(), "m"),
    )
    assert aux._try_configured_fallback_chain("compression", "anthropic") == (
        None, None, "",
    )
    # The "no client could be built" entry point delegates to the same helper,
    # so it is covered by the same switch.
    assert aux._try_configured_fallback_for_unavailable_client(
        "compression", "anthropic",
    ) == (None, None, "")


def test_configured_per_task_chain_still_works_when_allowed(allowed, monkeypatch):
    monkeypatch.setattr(
        aux, "_get_auxiliary_task_config",
        lambda task: {"fallback_chain": [{"provider": "openrouter", "model": "m"}]},
    )
    monkeypatch.setattr(
        aux, "_resolve_fallback_entry", lambda entry: (object(), "m"),
    )
    client, model, label = aux._try_configured_fallback_chain(
        "title_generation", "anthropic",
    )
    assert client is not None and model == "m" and "openrouter" in label


def test_main_agent_chain_is_suppressed(killed, monkeypatch):
    monkeypatch.setattr(
        aux, "_resolve_fallback_entry", lambda entry: (object(), "gpt-x"),
    )
    assert aux._try_main_fallback_chain("compression", "anthropic") == (None, None, "")
    # The config-level reader is gated too, so nothing downstream of it can
    # re-derive a chain from config.yaml either.
    assert get_fallback_chain(
        {"fallback_providers": [{"provider": "openai", "model": "gpt-x"}]}
    ) == []


def test_payment_and_builtin_discovery_are_suppressed(killed, monkeypatch):
    monkeypatch.setattr(aux, "_get_provider_chain", _sentinel_chain)
    assert aux._try_payment_fallback("anthropic", "compression") == (None, None, "")


def test_payment_discovery_still_works_when_allowed(allowed, monkeypatch):
    monkeypatch.setattr(aux, "_get_provider_chain", _sentinel_chain)
    monkeypatch.setattr(aux, "_read_main_provider", lambda: "anthropic")
    monkeypatch.setattr(aux, "_is_provider_unhealthy", lambda label: False)
    client, model, label = aux._try_payment_fallback("anthropic", "compression")
    assert client is not None and model == "some-model" and label == "openrouter"


def test_main_agent_model_safety_net_is_suppressed(killed, monkeypatch):
    monkeypatch.setattr(aux, "_read_main_provider", lambda: "openai")
    monkeypatch.setattr(aux, "_read_main_model", lambda: "gpt-x")
    monkeypatch.setattr(aux, "_is_provider_unhealthy", lambda label: False)
    monkeypatch.setattr(
        aux, "resolve_provider_client",
        lambda **kwargs: (object(), "gpt-x"),
    )
    assert aux._try_main_agent_model_fallback("anthropic", "compression") == (
        None, None, "",
    )


def test_main_agent_model_safety_net_still_works_when_allowed(allowed, monkeypatch):
    monkeypatch.setattr(aux, "_read_main_provider", lambda: "openai")
    monkeypatch.setattr(aux, "_read_main_model", lambda: "gpt-x")
    monkeypatch.setattr(aux, "_is_provider_unhealthy", lambda label: False)
    monkeypatch.setattr(
        aux, "resolve_provider_client",
        lambda **kwargs: (object(), "gpt-x"),
    )
    client, model, label = aux._try_main_agent_model_fallback(
        "anthropic", "compression",
    )
    assert client is not None and model == "gpt-x" and label == "main-agent(openai)"


def test_auto_route_does_not_substitute_after_the_main_provider_fails(
    killed, monkeypatch
):
    """Auto mode falls through Steps 2 and 3; both must stay closed."""
    monkeypatch.setattr(aux, "_get_provider_chain", _sentinel_chain)
    monkeypatch.setattr(aux, "_read_main_provider", lambda: "")
    monkeypatch.setattr(aux, "_read_main_model", lambda: "")
    monkeypatch.setattr(
        aux, "_get_auxiliary_task_config",
        lambda task: {"fallback_chain": [{"provider": "openrouter", "model": "m"}]},
    )
    monkeypatch.setattr(
        aux, "_resolve_fallback_entry", lambda entry: (object(), "m"),
    )
    assert aux._resolve_auto_route(main_runtime={}, task="compression") == (
        None, None, "",
    )


@pytest.mark.parametrize(
    "value,disabled",
    [
        ("1", True), ("true", True), ("TRUE", True), ("yes", True), ("on", True),
        ("0", False), ("", False), ("off", False), ("nope", False),
    ],
)
def test_switch_parsing(monkeypatch, value, disabled):
    monkeypatch.setenv(FALLBACKS_DISABLED_ENV, value)
    from hermes_cli.fallback_config import fallbacks_disabled

    assert fallbacks_disabled() is disabled


def test_the_cli_latch_outranks_any_environment_value(monkeypatch):
    """The argv channel cannot be undone by a later dotenv load.

    ``load_hermes_dotenv`` loads a profile's ``.env`` with ``override=True``,
    so an env-only kill switch is resettable. The latch is module state, which
    ``os.environ`` writes cannot reach.
    """
    from hermes_cli import fallback_config as fc

    monkeypatch.setattr(fc, "_fallbacks_disabled_for_process", False)
    monkeypatch.setenv(FALLBACKS_DISABLED_ENV, "0")
    assert fc.fallbacks_disabled() is False

    assert fc.apply_process_fallback_policy(["-p", "x", fc.NO_FALLBACK_FLAG]) is True
    # A profile .env "winning" afterwards changes nothing.
    monkeypatch.setenv(FALLBACKS_DISABLED_ENV, "0")
    assert fc.fallbacks_disabled() is True
    assert get_fallback_chain(
        {"fallback_providers": [{"provider": "openai", "model": "gpt-x"}]}
    ) == []
    assert fc.strip_no_fallback_flag(["-p", "x", fc.NO_FALLBACK_FLAG]) == ["-p", "x"]


def test_the_cli_latch_is_not_set_without_the_flag(monkeypatch):
    from hermes_cli import fallback_config as fc

    monkeypatch.setattr(fc, "_fallbacks_disabled_for_process", False)
    monkeypatch.delenv(FALLBACKS_DISABLED_ENV, raising=False)
    assert fc.apply_process_fallback_policy(["-p", "x", "chat"]) is False
    assert fc.fallbacks_disabled() is False


def test_the_real_cli_entry_point_latches_after_dotenv_loading(tmp_path):
    """Ordering regression, exercised through the REAL ``hermes_cli.main`` import.

    ``load_hermes_dotenv`` loads the profile's ``.env`` with ``override=True``.
    With the latch installed BEFORE that load, a ``.env`` saying
    ``HERMES_DISABLE_FALLBACKS=0`` wins over the env channel the latch
    re-exports, so anything this process spawns inherits "fallbacks allowed"
    even though this process is pinned. Importing the module is what runs both
    steps in their production order, so this asserts on their real effect
    rather than on the shape of the file.
    """
    import json
    import subprocess
    import sys

    home = tmp_path / "hermes"
    home.mkdir()
    home.joinpath(".env").write_text(
        f"{FALLBACKS_DISABLED_ENV}=0\n", encoding="utf-8"
    )
    child = (
        "import json, sys\n"
        "sys.argv = ['hermes', '--no-fallbacks', 'chat']\n"
        "import hermes_cli.main  # noqa: F401\n"
        "from hermes_cli.fallback_config import "
        "FALLBACKS_DISABLED_ENV, fallbacks_disabled\n"
        "import os\n"
        "print('RESULT' + json.dumps({\n"
        "    'disabled': fallbacks_disabled(),\n"
        "    'exported': os.environ.get(FALLBACKS_DISABLED_ENV),\n"
        "    'argv_clean': '--no-fallbacks' not in sys.argv,\n"
        "}))\n"
    )
    env = {
        **os.environ,
        "HERMES_HOME": str(home),
        "PYTHONPATH": str(_REPO_ROOT),
    }
    proc = subprocess.run(
        [sys.executable, "-c", child],
        capture_output=True, text=True, timeout=180, env=env, cwd=str(_REPO_ROOT),
    )
    assert proc.returncode == 0, proc.stderr
    line = next(
        row for row in proc.stdout.splitlines() if row.startswith("RESULT")
    )
    assert json.loads(line[len("RESULT"):]) == {
        "disabled": True,
        # The env channel a spawned child would inherit was NOT left at the
        # ``.env``'s 0 — which is only true when the latch runs afterwards.
        "exported": "1",
        "argv_clean": True,
    }


# ---------------------------------------------------------------------------
# A pinned run: exactly one admissible auxiliary route, on every surface
# ---------------------------------------------------------------------------
#
# Suppressing the fallback chains is only half the contract. An explicit
# argument, an ``auxiliary.<task>`` override, a caller-supplied reasoning
# control, an ``api_mode`` switch, the fast-model preference and the vision
# auto-detect chain each resolve a route BEFORE any fallback would be
# consulted, so each is a substitution the kill switch has to cover too.

_PINNED_PROVIDER = "anthropic"
_PINNED_MODEL = "claude-opus-5"
_PINNED_EFFORT = "max"
_PINNED_API_MODE = "anthropic_messages"
_PINNED_MESSAGES = [{"role": "user", "content": "summarize this"}]


class _RecordingClient:
    """A provider client that records every request it is asked to send."""

    base_url = "https://api.anthropic.com"

    def __init__(self):
        self.requests: list = []
        recorder = self

        class _Completions:
            def create(self, **kwargs):
                recorder.requests.append(kwargs)
                return {"ok": True}

        self.chat = SimpleNamespace(completions=_Completions())


class _AsyncRecordingClient(_RecordingClient):
    def __init__(self):
        super().__init__()
        recorder = self

        class _Completions:
            async def create(self, **kwargs):
                recorder.requests.append(kwargs)
                return {"ok": True}

        self.chat = SimpleNamespace(completions=_Completions())


@pytest.fixture
def pinned(monkeypatch):
    """A locked main runtime, installed exactly as ``build_turn_context`` does."""
    monkeypatch.setenv(FALLBACKS_DISABLED_ENV, "1")
    monkeypatch.setattr(aux, "_get_auxiliary_task_config", lambda task: {})
    monkeypatch.setattr(
        aux, "_validate_llm_response", lambda response, _task, **_kw: response,
    )
    token = aux.set_runtime_main(
        _PINNED_PROVIDER,
        _PINNED_MODEL,
        api_mode=_PINNED_API_MODE,
        reasoning_effort=_PINNED_EFFORT,
    )
    try:
        yield
    finally:
        aux.reset_runtime_main(token)


@pytest.fixture
def recorded(monkeypatch):
    """Install a recording client and capture how it was asked for."""
    sync_client = _RecordingClient()
    async_client = _AsyncRecordingClient()
    factory_calls: list = []

    def fake_cached_client(provider, model=None, **kwargs):
        factory_calls.append({"provider": provider, "model": model, **kwargs})
        client = async_client if kwargs.get("async_mode") else sync_client
        return client, model

    monkeypatch.setattr(aux, "_get_cached_client", fake_cached_client)
    return SimpleNamespace(
        sync=sync_client, asynchronous=async_client, factory_calls=factory_calls,
    )


def _sent(recorded, *, asynchronous=False):
    client = recorded.asynchronous if asynchronous else recorded.sync
    assert len(client.requests) == 1
    return client.requests[0], recorded.factory_calls[-1]


def test_a_pinned_sync_request_sends_the_exact_approved_route(pinned, recorded):
    aux.call_llm(task="compression", messages=_PINNED_MESSAGES)

    request, factory = _sent(recorded)
    assert (factory["provider"], factory["model"]) == (
        _PINNED_PROVIDER, _PINNED_MODEL,
    )
    assert factory["api_mode"] == _PINNED_API_MODE
    assert request["model"] == _PINNED_MODEL
    # The depth that actually goes on the wire, rebuilt from the approved route.
    assert request["_reasoning_config"] == {"enabled": True, "effort": _PINNED_EFFORT}


@pytest.mark.asyncio
async def test_a_pinned_async_request_sends_the_exact_approved_route(
    pinned, recorded
):
    await aux.async_call_llm(task="compression", messages=_PINNED_MESSAGES)

    request, factory = _sent(recorded, asynchronous=True)
    assert (factory["provider"], factory["model"]) == (
        _PINNED_PROVIDER, _PINNED_MODEL,
    )
    assert factory["api_mode"] == _PINNED_API_MODE
    assert request["model"] == _PINNED_MODEL
    assert request["_reasoning_config"] == {"enabled": True, "effort": _PINNED_EFFORT}


@pytest.mark.parametrize(
    "reasoning_config",
    [
        {"enabled": True, "effort": "low"},
        {"effort": "medium"},
        # An explicit "do not reason" is as much a depth change as naming one.
        {"enabled": False},
        # Never admissible at all, whatever the route resolved to.
        {"enabled": True, "effort": "ultra"},
        # A reasoning object with no recognizable depth field still reshapes
        # the request.
        {"unknown_control": 1},
    ],
)
def test_a_conflicting_reasoning_config_is_refused_before_any_client_call(
    pinned, recorded, reasoning_config
):
    with pytest.raises(aux.AuxiliaryRouteLocked):
        aux.call_llm(
            task="compression",
            messages=_PINNED_MESSAGES,
            reasoning_config=reasoning_config,
        )

    assert recorded.sync.requests == []
    assert recorded.factory_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reasoning_config", [{"effort": "low"}, {"enabled": False}, {"effort": "ultra"}],
)
async def test_an_async_conflicting_reasoning_config_is_refused_before_any_call(
    pinned, recorded, reasoning_config
):
    with pytest.raises(aux.AuxiliaryRouteLocked):
        await aux.async_call_llm(
            task="compression",
            messages=_PINNED_MESSAGES,
            reasoning_config=reasoning_config,
        )

    assert recorded.asynchronous.requests == []
    assert recorded.factory_calls == []


@pytest.mark.parametrize(
    "extra_body",
    [
        {"reasoning": {"effort": "low"}},
        {"reasoning_effort": "low"},
        {"thinking": {"budget_tokens": 128}},
        {"thinking_config": {"effort": "minimal"}},
        {"reasoning_config": {"effort": "ultra"}},
        {"reasoning": {"enabled": False}},
    ],
)
def test_a_nested_extra_body_effort_is_refused_before_any_client_call(
    pinned, recorded, extra_body
):
    with pytest.raises(aux.AuxiliaryRouteLocked):
        aux.call_llm(
            task="compression", messages=_PINNED_MESSAGES, extra_body=extra_body,
        )

    assert recorded.sync.requests == []
    assert recorded.factory_calls == []


def test_a_conflicting_api_mode_is_refused_before_any_client_call(pinned, recorded):
    with pytest.raises(aux.AuxiliaryRouteLocked):
        aux.call_llm(
            task="compression",
            messages=_PINNED_MESSAGES,
            api_mode="chat_completions",
        )

    assert recorded.sync.requests == []
    assert recorded.factory_calls == []


def test_exactly_matching_caller_controls_are_accepted_and_still_send_the_pin(
    pinned, recorded
):
    """A caller that faithfully echoes the approved route still works.

    And the transport-only ``extra_body`` fields it sent — which carry no
    approval authority — survive, while the request's depth comes from the pin.
    """
    aux.call_llm(
        task="compression",
        messages=_PINNED_MESSAGES,
        provider=_PINNED_PROVIDER,
        model=_PINNED_MODEL,
        api_mode=_PINNED_API_MODE,
        reasoning_config={"enabled": True, "effort": _PINNED_EFFORT},
        extra_body={"session_id": "sticky-1", "reasoning": {"effort": _PINNED_EFFORT}},
    )

    request, factory = _sent(recorded)
    assert factory["api_mode"] == _PINNED_API_MODE
    assert request["model"] == _PINNED_MODEL
    assert request["_reasoning_config"] == {"enabled": True, "effort": _PINNED_EFFORT}
    assert request["extra_body"]["session_id"] == "sticky-1"


@pytest.mark.parametrize(
    "override",
    [
        {"provider": "openai-codex"},
        {"model": "gpt-5.6-sol"},
        {"base_url": "https://elsewhere.example/v1"},
    ],
)
def test_a_caller_named_route_off_the_pin_is_refused(pinned, recorded, override):
    with pytest.raises(aux.AuxiliaryRouteLocked):
        aux.call_llm(task="compression", messages=_PINNED_MESSAGES, **override)

    assert recorded.factory_calls == []


def test_a_conflicting_auxiliary_task_configuration_is_refused(
    pinned, recorded, monkeypatch
):
    """An ``auxiliary.<task>`` entry resolves before any fallback would."""
    monkeypatch.setattr(
        aux, "_get_auxiliary_task_config",
        lambda task: {"provider": "openai-codex", "model": "gpt-5.6-sol"},
    )

    with pytest.raises(aux.AuxiliaryRouteLocked):
        aux.call_llm(task="compression", messages=_PINNED_MESSAGES)

    assert recorded.factory_calls == []


def test_a_per_task_reasoning_override_cannot_change_the_approved_depth(
    pinned, recorded, monkeypatch
):
    monkeypatch.setattr(
        aux, "_get_auxiliary_task_config",
        lambda task: {"reasoning_effort": "low"},
    )

    with pytest.raises(aux.AuxiliaryRouteLocked):
        aux.call_llm(task="compression", messages=_PINNED_MESSAGES)

    assert recorded.factory_calls == []


def test_a_per_task_extra_body_reasoning_field_is_dropped_on_a_pinned_run(
    pinned, monkeypatch
):
    """The wire-level reasoning controls never survive into the request body."""
    monkeypatch.setattr(
        aux, "_get_auxiliary_task_config",
        lambda task: {"extra_body": {"reasoning": {"effort": "max"}, "tags": ["t"]}},
    )

    body = aux._get_task_extra_body("compression")

    assert body == {"tags": ["t"]}


def test_an_ultra_approved_effort_is_never_admissible(monkeypatch, recorded):
    monkeypatch.setenv(FALLBACKS_DISABLED_ENV, "1")
    token = aux.set_runtime_main(
        _PINNED_PROVIDER, _PINNED_MODEL, reasoning_effort="ultra",
    )
    try:
        with pytest.raises(aux.AuxiliaryRouteLocked):
            aux.call_llm(task="compression", messages=_PINNED_MESSAGES)
    finally:
        aux.reset_runtime_main(token)

    assert recorded.factory_calls == []


def test_an_unresolvable_pin_refuses_rather_than_falling_through(
    monkeypatch, recorded
):
    """Silence here would mean ordinary resolution — the exact substitution."""
    monkeypatch.setenv(FALLBACKS_DISABLED_ENV, "1")
    monkeypatch.setattr(aux, "_read_main_provider", lambda: "")
    monkeypatch.setattr(aux, "_read_main_model", lambda: "")
    token = aux.set_runtime_main("", "")
    try:
        with pytest.raises(aux.AuxiliaryRouteLocked):
            aux.call_llm(task="compression", messages=_PINNED_MESSAGES)
    finally:
        aux.reset_runtime_main(token)

    assert recorded.factory_calls == []


def test_a_pinned_route_that_cannot_be_built_is_never_substituted(
    pinned, monkeypatch
):
    monkeypatch.setattr(
        aux, "_get_cached_client", lambda *args, **kwargs: (None, None),
    )
    monkeypatch.setattr(
        aux, "_try_configured_fallback_for_unavailable_client",
        lambda task, provider: (object(), "substitute", "openrouter"),
    )

    with pytest.raises(aux.AuxiliaryRouteLocked):
        aux.call_llm(task="compression", messages=_PINNED_MESSAGES)


def test_a_pinned_run_never_prefers_the_fast_model(monkeypatch):
    """Swapping in the provider's cheap model is a substitution like any other."""
    monkeypatch.setattr(
        aux, "_get_auxiliary_task_config", lambda task: {"prefer_fast_model": True},
    )
    monkeypatch.delenv(FALLBACKS_DISABLED_ENV, raising=False)
    assert aux._task_prefers_fast_model("title_generation") is True

    monkeypatch.setenv(FALLBACKS_DISABLED_ENV, "1")
    token = aux.set_runtime_main(
        _PINNED_PROVIDER, _PINNED_MODEL, reasoning_effort=_PINNED_EFFORT,
    )
    try:
        assert aux._task_prefers_fast_model("title_generation") is False
    finally:
        aux.reset_runtime_main(token)


def test_pinned_vision_uses_the_approved_route_and_never_auto_detects(
    pinned, monkeypatch
):
    client = _RecordingClient()
    seen: list = []

    def fake_resolve(provider, **kwargs):
        seen.append({"provider": provider, **kwargs})
        return client, kwargs.get("model")

    monkeypatch.setattr(aux, "resolve_provider_client", fake_resolve)
    monkeypatch.setattr(
        aux, "_normalize_vision_provider",
        lambda _value: pytest.fail("the vision auto-detect chain was consulted"),
    )

    resolved = aux.resolve_vision_provider_client(provider="auto")

    assert resolved == (_PINNED_PROVIDER, client, _PINNED_MODEL)
    assert seen[-1]["provider"] == _PINNED_PROVIDER
    assert seen[-1]["model"] == _PINNED_MODEL
    assert seen[-1]["api_mode"] == _PINNED_API_MODE


def test_pinned_vision_refuses_a_backend_named_off_the_pin(pinned, monkeypatch):
    monkeypatch.setattr(
        aux, "resolve_provider_client",
        lambda *args, **kwargs: pytest.fail("a substitute vision client was built"),
    )

    with pytest.raises(aux.AuxiliaryRouteLocked):
        aux.resolve_vision_provider_client(
            provider="openai-codex", model="gpt-5.6-sol",
        )


def test_pinned_vision_refuses_when_the_approved_client_cannot_be_built(
    pinned, monkeypatch
):
    monkeypatch.setattr(
        aux, "resolve_provider_client", lambda *args, **kwargs: (None, None),
    )

    with pytest.raises(aux.AuxiliaryRouteLocked):
        aux.resolve_vision_provider_client(provider="auto")


@pytest.mark.parametrize("factory", ["sync", "async"])
def test_the_text_auxiliary_client_factories_resolve_the_pinned_route(
    pinned, monkeypatch, factory
):
    """Both entry points now thread ``main_runtime`` into route resolution."""
    seen: list = []

    def fake_resolve(provider, **kwargs):
        seen.append({"provider": provider, **kwargs})
        return object(), kwargs.get("model")

    monkeypatch.setattr(aux, "resolve_provider_client", fake_resolve)

    if factory == "sync":
        aux.get_text_auxiliary_client(task="compression")
    else:
        aux.get_async_text_auxiliary_client(task="compression")

    assert seen[-1]["provider"] == _PINNED_PROVIDER
    assert seen[-1]["model"] == _PINNED_MODEL
    assert seen[-1]["api_mode"] == _PINNED_API_MODE


def test_an_unpinned_run_keeps_its_ordinary_auxiliary_behaviour(
    allowed, recorded, monkeypatch
):
    """The whole lock is inert when this process is not pinned."""
    monkeypatch.setattr(
        aux, "_validate_llm_response", lambda response, _task, **_kw: response,
    )
    monkeypatch.setattr(aux, "_get_auxiliary_task_config", lambda task: {})
    token = aux.set_runtime_main(
        _PINNED_PROVIDER, _PINNED_MODEL, reasoning_effort=_PINNED_EFFORT,
    )
    try:
        aux.call_llm(
            task="compression",
            messages=_PINNED_MESSAGES,
            provider="openai-codex",
            model="gpt-5.6-sol",
            reasoning_config={"enabled": True, "effort": "low"},
        )
    finally:
        aux.reset_runtime_main(token)

    request, factory = _sent(recorded)
    assert factory["provider"] == "openai-codex"
    assert request["model"] == "gpt-5.6-sol"


# ---------------------------------------------------------------------------
# Item 32TK: the pinned MODEL survives the request body and every recovery
# ---------------------------------------------------------------------------
#
# Pinning the client is not the same as pinning the wire request. An
# OpenAI-compatible client merges ``extra_body`` verbatim into the body, and
# every "self-heal" branch below the first failure re-resolves a model. Each is
# a way for a pinned run to send a route nobody approved while the audit still
# reports the approved one.


@pytest.mark.parametrize(
    "off_route_body",
    [
        {"model": "gpt-5.6-sol"},
        {"provider": {"order": ["openai"]}},
        {"base_url": "https://elsewhere.example/v1"},
        {"messages": [{"role": "user", "content": "something else"}]},
    ],
)
def test_a_caller_extra_body_off_the_allowlist_is_refused(
    pinned, recorded, off_route_body
):
    with pytest.raises(aux.AuxiliaryRouteLocked):
        aux.call_llm(
            task="compression", messages=_PINNED_MESSAGES, extra_body=off_route_body,
        )

    assert recorded.sync.requests == []
    assert recorded.factory_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("off_route_body", [{"model": "gpt-5.6-sol"}, {"top_p": 0.1}])
async def test_an_async_caller_extra_body_off_the_allowlist_is_refused(
    pinned, recorded, off_route_body
):
    with pytest.raises(aux.AuxiliaryRouteLocked):
        await aux.async_call_llm(
            task="compression", messages=_PINNED_MESSAGES, extra_body=off_route_body,
        )

    assert recorded.asynchronous.requests == []
    assert recorded.factory_calls == []


@pytest.mark.parametrize("caller", ["sync", "async"])
def test_a_task_configured_extra_body_cannot_re_add_a_model(
    pinned, monkeypatch, caller
):
    """``auxiliary.<task>.extra_body`` is the request body's BASE, so a model
    named there would outrank the pin on the wire."""
    monkeypatch.setattr(
        aux, "_get_auxiliary_task_config",
        lambda task: {"extra_body": {"model": "gpt-5.6-sol", "tags": ["t"]}},
    )

    # The filter itself: nothing off the allowlist survives into the body.
    assert aux._get_task_extra_body("compression") == {"tags": ["t"]}

    # And the request is refused outright rather than quietly sanitized.
    with pytest.raises(aux.AuxiliaryRouteLocked):
        if caller == "sync":
            aux.call_llm(task="compression", messages=_PINNED_MESSAGES)
        else:
            import asyncio

            asyncio.run(
                aux.async_call_llm(task="compression", messages=_PINNED_MESSAGES)
            )


def test_an_unpinned_run_still_sends_its_whole_extra_body(allowed, recorded, monkeypatch):
    """The allowlist is inert when this process is not pinned."""
    monkeypatch.setattr(
        aux, "_validate_llm_response", lambda response, _task, **_kw: response,
    )
    monkeypatch.setattr(
        aux, "_get_auxiliary_task_config",
        lambda task: {"extra_body": {"enable_thinking": False}},
    )
    aux.call_llm(
        task="compression",
        messages=_PINNED_MESSAGES,
        provider=_PINNED_PROVIDER,
        model=_PINNED_MODEL,
        extra_body={"top_p": 0.5},
    )

    request, _factory = _sent(recorded)
    assert request["extra_body"] == {"enable_thinking": False, "top_p": 0.5}


class _NotFound(Exception):
    status_code = 404

    def __str__(self):
        return "Model 'claude-opus-5' not found. The requested model does not exist."


class _Unauthorized(Exception):
    status_code = 401


class _PaymentRequired(Exception):
    status_code = 402


@pytest.fixture
def failing(monkeypatch):
    """A pinned Nous route whose first request always fails."""
    monkeypatch.setenv(FALLBACKS_DISABLED_ENV, "1")
    monkeypatch.setattr(aux, "_get_auxiliary_task_config", lambda task: {})
    monkeypatch.setattr(
        aux, "_validate_llm_response", lambda response, _task, **_kw: response,
    )
    monkeypatch.setattr(
        aux, "_get_cached_client",
        lambda provider, model=None, **kwargs: (SimpleNamespace(
            base_url="https://inference-api.nousresearch.com/v1",
        ), model),
    )
    token = aux.set_runtime_main("nous", "Hermes-4-405B", reasoning_effort="max")
    try:
        yield SimpleNamespace(sent=[])
    finally:
        aux.reset_runtime_main(token)


def _install_relay(monkeypatch, failing, error, *, asynchronous=False):
    """Make every relay attempt raise ``error`` and record its kwargs."""
    if asynchronous:
        async def relay(client, kwargs, **_ignored):
            failing.sent.append(dict(kwargs))
            raise error
        monkeypatch.setattr(aux, "_relay_async_completion", relay)
    else:
        def relay(client, kwargs, **_ignored):
            failing.sent.append(dict(kwargs))
            raise error
        monkeypatch.setattr(aux, "_relay_sync_completion", relay)


def test_a_pinned_run_never_heals_onto_a_different_model(failing, monkeypatch):
    """The pinned model 404ing means the approved route is unavailable — every
    recommendation this heal could reach is a different model."""
    _install_relay(monkeypatch, failing, _NotFound())
    monkeypatch.setattr(
        aux, "_refresh_nous_recommended_model",
        lambda **kwargs: pytest.fail("a substitute model was resolved"),
    )

    with pytest.raises(aux.AuxiliaryRouteLocked):
        aux.call_llm(task="compression", messages=_PINNED_MESSAGES)

    assert [request["model"] for request in failing.sent] == ["Hermes-4-405B"]


@pytest.mark.asyncio
async def test_an_async_pinned_run_never_heals_onto_a_different_model(
    failing, monkeypatch
):
    _install_relay(monkeypatch, failing, _NotFound(), asynchronous=True)
    monkeypatch.setattr(
        aux, "_refresh_nous_recommended_model",
        lambda **kwargs: pytest.fail("a substitute model was resolved"),
    )

    with pytest.raises(aux.AuxiliaryRouteLocked):
        await aux.async_call_llm(task="compression", messages=_PINNED_MESSAGES)

    assert [request["model"] for request in failing.sent] == ["Hermes-4-405B"]


@pytest.mark.parametrize("error", [_Unauthorized(), _PaymentRequired()])
def test_a_pinned_credential_refresh_never_swaps_the_model(
    failing, monkeypatch, error
):
    """Refreshing credentials is legitimate under the pin; taking the refreshed
    route's model with them is the substitution."""
    _install_relay(monkeypatch, failing, error)
    monkeypatch.setattr(
        aux, "_nous_portal_account_has_fresh_paid_access", lambda: True,
    )
    monkeypatch.setattr(
        aux, "_refresh_nous_auxiliary_client",
        lambda **kwargs: (object(), "Hermes-4-70B"),
    )

    with pytest.raises(aux.AuxiliaryRouteLocked):
        aux.call_llm(task="compression", messages=_PINNED_MESSAGES)

    assert [request["model"] for request in failing.sent] == ["Hermes-4-405B"]


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [_Unauthorized(), _PaymentRequired()])
async def test_an_async_pinned_credential_refresh_never_swaps_the_model(
    failing, monkeypatch, error
):
    _install_relay(monkeypatch, failing, error, asynchronous=True)
    monkeypatch.setattr(
        aux, "_nous_portal_account_has_fresh_paid_access", lambda: True,
    )
    monkeypatch.setattr(
        aux, "_refresh_nous_auxiliary_client",
        lambda **kwargs: (object(), "Hermes-4-70B"),
    )

    with pytest.raises(aux.AuxiliaryRouteLocked):
        await aux.async_call_llm(task="compression", messages=_PINNED_MESSAGES)

    assert [request["model"] for request in failing.sent] == ["Hermes-4-405B"]


def test_a_pinned_credential_refresh_that_keeps_the_model_still_retries(
    failing, monkeypatch
):
    """The adjacent success path: same model, fresh credentials, one retry."""
    attempts: list = []

    def relay(client, kwargs, **_ignored):
        attempts.append(dict(kwargs))
        if len(attempts) == 1:
            raise _Unauthorized()
        return {"ok": True}

    monkeypatch.setattr(aux, "_relay_sync_completion", relay)
    monkeypatch.setattr(
        aux, "_refresh_nous_auxiliary_client",
        lambda **kwargs: (object(), "Hermes-4-405B"),
    )

    assert aux.call_llm(task="compression", messages=_PINNED_MESSAGES) == {"ok": True}
    assert [request["model"] for request in attempts] == [
        "Hermes-4-405B", "Hermes-4-405B",
    ]
