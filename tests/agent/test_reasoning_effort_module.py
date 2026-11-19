"""Tests for agent.reasoning_effort — the canonical effort ladder + clamp.

This module is the single source of truth every transport and provider
profile uses to translate Hermes' internal effort ladder onto a wire's
supported vocabulary. The policy under test:

- supported levels pass through verbatim
- unsupported levels clamp to the nearest WEAKER supported level (never
  escalate above the ask, never invert the ladder)
- nothing weaker → weakest supported level
- "none" is never a degradation target (would silently disable thinking)
- declared overrides (vendor-documented roundings) win over nearest-weaker
- unknown supported-sets and bespoke level names pass through unchanged
"""

import pytest

from agent.reasoning_effort import (
    CODEX_GPT56_EFFORTS,
    EFFORT_LADDER,
    GLM52_EFFORTS,
    GLM52_OVERRIDES,
    KIMI_K2_EFFORTS,
    KIMI_K3_EFFORTS,
    KIMI_K3_OVERRIDES,
    OPENAI_COMPAT_WIRE_EFFORTS,
    clamp_effort,
    kimi_supported_efforts,
    requested_effort,
)
from hermes_constants import VALID_REASONING_EFFORTS


class TestLadderContract:
    def test_ladder_is_superset_of_valid_efforts(self):
        """The canonical ladder must cover every level users can configure —
        an internal level missing from the ladder would pass through clamps
        unchanged and leak to the wire (the #89503 class)."""
        for level in VALID_REASONING_EFFORTS:
            assert level in EFFORT_LADDER, level

    def test_no_declared_wire_set_contains_ultra(self):
        """ultra is internal vocabulary; every wire set must exclude it so it
        always clamps down."""
        import agent.reasoning_effort as mod

        for name in dir(mod):
            if name.endswith("_EFFORTS"):
                assert "ultra" not in getattr(mod, name), name


class TestClampEffort:
    def test_supported_levels_pass_through(self):
        for level in OPENAI_COMPAT_WIRE_EFFORTS:
            assert clamp_effort(level, OPENAI_COMPAT_WIRE_EFFORTS) == level

    def test_ultra_clamps_to_max_on_openai_wire(self):
        assert clamp_effort("ultra", OPENAI_COMPAT_WIRE_EFFORTS) == "max"

    def test_nearest_weaker_never_escalates(self):
        # xhigh against low..high → high (weaker), never max.
        assert clamp_effort("xhigh", ("low", "medium", "high", "max")) == "high"

    def test_floor_when_nothing_weaker(self):
        assert clamp_effort("minimal", ("low", "medium")) == "low"

    def test_none_is_never_a_degradation_target(self):
        # minimal against {none, low}: clamping to none would silently
        # disable thinking — must take low.
        assert clamp_effort("minimal", ("none", "low", "high")) == "low"

    def test_none_still_passes_through_when_requested(self):
        assert clamp_effort("none", ("none", "low", "high")) == "none"

    def test_unknown_supported_set_passes_through(self):
        assert clamp_effort("ultra", None) == "ultra"
        assert clamp_effort("ultra", []) == "ultra"

    def test_bespoke_level_passes_through(self):
        assert clamp_effort("turbo-9000", ("low", "high")) == "turbo-9000"

    def test_empty_effort_passes_through(self):
        assert clamp_effort(None, ("low",)) is None
        assert clamp_effort("", ("low",)) == ""

    def test_overrides_win(self):
        assert clamp_effort("medium", KIMI_K3_EFFORTS, KIMI_K3_OVERRIDES) == "high"
        assert clamp_effort("xhigh", KIMI_K3_EFFORTS, KIMI_K3_OVERRIDES) == "max"

    def test_override_ignored_when_target_unsupported(self):
        # An override pointing outside the supported set falls back to the
        # ladder walk instead of emitting an invalid level.
        assert clamp_effort("medium", ("low", "high"), {"medium": "max"}) == "low"

    def test_monotonic_over_full_ladder(self):
        """A stronger ask never resolves weaker than a weaker ask — for every
        declared wire set."""
        import agent.reasoning_effort as mod

        sets = [
            getattr(mod, name) for name in dir(mod) if name.endswith("_EFFORTS")
        ]
        enabled_ladder = [l for l in EFFORT_LADDER if l != "none"]
        for supported in sets:
            prev_rank = -1
            for level in enabled_ladder:
                out = clamp_effort(level, supported)
                rank = EFFORT_LADDER.index(out)
                assert rank >= prev_rank, (supported, level, out)
                prev_rank = rank


class TestKimiVocabulary:
    @pytest.mark.parametrize(
        "model",
        ["k3", "kimi-k3", "kimi-k3-cot", "moonshotai/kimi-k3", "k3-256k"],
    )
    def test_k3_slugs(self, model):
        assert kimi_supported_efforts(model) is KIMI_K3_EFFORTS

    @pytest.mark.parametrize(
        "model",
        ["kimi-k2.6", "moonshotai/kimi-k2-0905", "kimi-latest", "mk3000", None],
    )
    def test_k2_era_slugs(self, model):
        assert kimi_supported_efforts(model) is KIMI_K2_EFFORTS


class TestGlm52Vocabulary:
    def test_two_level_set(self):
        assert clamp_effort("ultra", GLM52_EFFORTS, GLM52_OVERRIDES) == "max"
        assert clamp_effort("xhigh", GLM52_EFFORTS, GLM52_OVERRIDES) == "max"
        # GLM's floor is high — weaker asks land there.
        assert clamp_effort("low", GLM52_EFFORTS, GLM52_OVERRIDES) == "high"
        assert clamp_effort("medium", GLM52_EFFORTS, GLM52_OVERRIDES) == "high"


class TestCodexVocabulary:
    def test_minimal_and_ultra(self):
        assert clamp_effort("minimal", CODEX_GPT56_EFFORTS) == "low"
        assert clamp_effort("ultra", CODEX_GPT56_EFFORTS) == "max"

    def test_per_model_max_support(self):
        """Live-verified (Aug 2026, #68365): 'max' is gpt-5.6-only — gpt-5.5
        rejects it ("Supported values are: 'none','low','medium','high',
        'xhigh'"); 'minimal' is rejected by both generations."""
        from agent.reasoning_effort import (
            CODEX_GPT56_EFFORTS,
            CODEX_LEGACY_EFFORTS,
            codex_supported_efforts,
        )

        assert codex_supported_efforts("gpt-5.6") is CODEX_GPT56_EFFORTS
        assert codex_supported_efforts("gpt-5.6-codex") is CODEX_GPT56_EFFORTS
        assert codex_supported_efforts("gpt-5.5") is CODEX_LEGACY_EFFORTS
        assert codex_supported_efforts("o5-pro") is CODEX_LEGACY_EFFORTS
        # The consequential clamps:
        assert clamp_effort("max", CODEX_GPT56_EFFORTS) == "max"
        assert clamp_effort("max", CODEX_LEGACY_EFFORTS) == "xhigh"
        assert clamp_effort("ultra", CODEX_LEGACY_EFFORTS) == "xhigh"
        assert clamp_effort("minimal", CODEX_LEGACY_EFFORTS) == "low"

    @pytest.mark.parametrize(
        "model", ["gpt-6-sol", "openai/gpt-6-sol", "GPT-6-Sol", "gpt-6-sol-900k"]
    )
    def test_gpt6_sol_keeps_max_instead_of_silently_clamping(self, model):
        """GPT-6 Sol speaks the gpt-5.6 vocabulary. On the legacy set a
        requested 'max' would silently become 'xhigh' — a reasoning downgrade."""
        from agent.reasoning_effort import codex_supported_efforts

        assert codex_supported_efforts(model) is CODEX_GPT56_EFFORTS
        assert clamp_effort("max", codex_supported_efforts(model)) == "max"
        assert clamp_effort("xhigh", codex_supported_efforts(model)) == "xhigh"
        # Never-escalate still holds: ultra caps at max, minimal floors at low.
        assert clamp_effort("ultra", codex_supported_efforts(model)) == "max"
        assert clamp_effort("minimal", codex_supported_efforts(model)) == "low"

    @pytest.mark.parametrize(
        "model", ["gpt-6-astra", "gpt-6-astra-900k", "openai/gpt-6-astra"]
    )
    def test_astra_keeps_its_own_vocabulary(self, model):
        """Astra is resolved before any generation marker and is unchanged:
        no disable level, max supported."""
        from agent.reasoning_effort import (
            CODEX_ASTRA_EFFORTS,
            codex_supported_efforts,
            is_astra_model,
        )

        assert CODEX_ASTRA_EFFORTS == ("low", "medium", "high", "xhigh", "max")
        assert is_astra_model(model) is True
        assert codex_supported_efforts(model) is CODEX_ASTRA_EFFORTS
        assert clamp_effort("xhigh", codex_supported_efforts(model)) == "xhigh"
        assert clamp_effort("max", codex_supported_efforts(model)) == "max"
        assert is_astra_model("gpt-6-sol") is False

    @pytest.mark.parametrize(
        "model",
        ["gpt-6.1-sol", "openai/gpt-6.1-sol", "GPT-6.1-Sol", "gpt-6.2-sol", "openai/gpt-6.2-sol"],
    )
    def test_gpt61_sol_keeps_max_and_has_no_disable_level(self, model):
        """GPT-6.1 Sol's model page (2026-09-29) lists low, medium, high, xhigh
        and max, and rejects none and minimal: max stays max, and a disable
        request floors at low instead of reaching the wire. GPT-6.2 Sol speaks
        the same set (OpenAI developer documentation, read 2026-11-19)."""
        from agent.reasoning_effort import codex_supported_efforts, is_astra_model

        assert codex_supported_efforts(model) == ("low", "medium", "high", "xhigh", "max")
        assert clamp_effort("max", codex_supported_efforts(model)) == "max"
        assert clamp_effort("none", codex_supported_efforts(model)) == "low"
        assert clamp_effort("minimal", codex_supported_efforts(model)) == "low"
        assert is_astra_model(model) is False

    @pytest.mark.parametrize("model", ["gpt-6.2-sol", "openai/gpt-6.2-sol"])
    def test_gpt62_sol_gets_the_five_levels_with_max_and_no_none(self, model):
        """GPT-6.2 Sol accepts low, medium, high, xhigh and max and rejects none
        (OpenAI developer documentation, read 2026-11-19). On the legacy set the
        monthly model check refused it for a role that runs at max."""
        from agent.reasoning_effort import codex_supported_efforts

        efforts = codex_supported_efforts(model)
        assert efforts == ("low", "medium", "high", "xhigh", "max")
        assert "max" in efforts and "none" not in efforts

    @pytest.mark.parametrize(
        "model", ["gpt-5.5", "gpt-5.4-mini", "gpt-5.3-codex", "gpt-6-astra-pro", None]
    )
    def test_older_and_unverified_codex_models_stay_legacy(self, model):
        """Only the verified generations gain max; an id that merely shares
        the gpt-6 prefix is not rerouted."""
        from agent.reasoning_effort import CODEX_LEGACY_EFFORTS, codex_supported_efforts

        assert codex_supported_efforts(model) is CODEX_LEGACY_EFFORTS
        assert clamp_effort("max", codex_supported_efforts(model)) == "xhigh"

    def test_unlisted_sol_version_keeps_the_legacy_set(self):
        """An id that no list names keeps the legacy set: the table stays
        explicit until a later card teaches the platform a new version
        without a code edit."""
        from agent.reasoning_effort import CODEX_LEGACY_EFFORTS, codex_supported_efforts

        assert codex_supported_efforts("gpt-6.3-sol") is CODEX_LEGACY_EFFORTS

    def test_a_learned_record_leaves_the_name_rule_as_it_is(self):
        """The learned effort record (model_efforts.json at the root home) teaches the Codex backend
        request, never this rule: each model gets the same answer by name while the record's entries
        govern that route's final request."""
        import json
        import os
        from pathlib import Path

        import httpx
        import openai

        import agent.transports.codex  # noqa: F401
        from agent.reasoning_effort import codex_supported_efforts
        from agent.transports import get_transport

        models = ("gpt-5.5", "gpt-5.6", "gpt-6-astra", "gpt-6.2-sol", "gpt-6.3-sol")
        by_name = {model: codex_supported_efforts(model) for model in models}
        entry = {
            "efforts": ["none", "low", "medium", "high", "xhigh", "max"], "refused": [],
            "thinking": {"adaptive": None, "manual": None, "disable": True},
            "source": "account_list", "observed_at": 1795070000,
        }
        record = {"schema_version": 1, "models": {f"openai-codex:{model}": entry for model in models}}
        (Path(os.environ["HERMES_HOME"]) / "model_efforts.json").write_text(json.dumps(record), encoding="utf-8")

        assert {model: codex_supported_efforts(model) for model in models} == by_name
        transport = get_transport("codex_responses")
        kwargs = transport.build_kwargs(
            model="gpt-6.3-sol", messages=[{"role": "user", "content": "Hi"}], tools=[], provider="openai-codex",
            base_url="https://chatgpt.com/backend-api/codex", is_codex_backend=True,
            reasoning_config={"enabled": True, "effort": "max"},
        )
        sent = []

        def capture(request):
            sent.append(json.loads(request.content))
            return httpx.Response(400, json={"error": {"message": "offline", "type": "invalid_request_error"}})

        client = openai.OpenAI(
            api_key="placeholder", base_url="https://chatgpt.com/backend-api/codex", max_retries=0,
            http_client=httpx.Client(transport=httpx.MockTransport(capture)),
        )
        with pytest.raises(openai.APIStatusError):
            client.responses.create(**transport.preflight_kwargs(kwargs))
        assert sent[0]["reasoning"]["effort"] == "max"  # the serialized body, after the preflight and the SDK
        assert kwargs["reasoning"]["effort"] == "max"  # the record governs the request; by name it is xhigh


class TestRequestedEffort:
    def test_extracts_effort(self):
        assert requested_effort({"enabled": True, "effort": "High"}) == "high"

    def test_none_when_absent_or_disabled(self):
        assert requested_effort(None) is None
        assert requested_effort({}) is None
        assert requested_effort({"enabled": False, "effort": "high"}) is None
        assert requested_effort("not-a-dict") is None
        assert requested_effort({"effort": ""}) is None
