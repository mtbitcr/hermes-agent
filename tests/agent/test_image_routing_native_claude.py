"""The main route's vision verdict in agent/image_routing.py.

A ``claude-`` model on the ``anthropic`` provider sees pictures even before the models.dev
catalog lists it; an explicit catalog False still wins, and other providers get no verdict from
this probe.
"""

from __future__ import annotations

from unittest.mock import patch

from agent import image_routing
from agent.models_dev import ModelCapabilities

MODEL = "claude-sonnet-5-5"


def test_a_claude_model_the_catalog_does_not_list_sees_pictures():
    with patch("agent.models_dev.get_model_capabilities", return_value=None):
        assert image_routing._lookup_supports_vision("anthropic", MODEL, {}) is True


def test_an_explicit_catalog_false_still_wins():
    with patch("agent.models_dev.get_model_capabilities", return_value=ModelCapabilities(supports_vision=False)):
        assert image_routing._lookup_supports_vision("anthropic", MODEL, {}) is False


def test_another_provider_with_a_claude_model_name_gets_no_verdict():
    assert image_routing._probe_native_claude("openrouter", MODEL, {}) is None
