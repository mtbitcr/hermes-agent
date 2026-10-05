"""Models examples (contract map section 5, test 8).

The options payload and the OAuth provider list are the two Models machine
projections, called on fixture input as Section B of the map says: native
option rows for a role assigned on one admitted provider and not the other,
with the route revision of that role's home, and a native provider list with
a Claude Code login. Model info and the refused options read come from the
real web_server handlers, called in process with the Models machine request of
tests/plugins/dashboard_auth/test_raphael_model_policy.py. Its codex_accounts
fixture writes both role homes and stubs the success-path audit, which a
stand-in request cannot pass. What is asserted is described in
tests/contracts/conftest.py.
"""

from __future__ import annotations

import json

import pytest
from fastapi import HTTPException
from fastapi.exception_handlers import http_exception_handler

from hermes_cli import web_server
from hermes_cli.profiles import get_profile_dir
from hermes_cli.web_routers import profiles as profile_routes
from plugins.dashboard_auth.raphael_workspace import model_policy
from tests.contracts import conftest as contract
from tests.plugins.dashboard_auth.test_raphael_model_policy import (  # noqa: F401  (a fixture)
    _options_request,
    codex_accounts,
)

FAMILY = "models"
# The builder has an assignment on Anthropic and none on OpenAI. codex_accounts
# writes each home on GPT-6.1 Sol, the verifier's own OpenAI route.
OPTIONS_ROLE = "raphael-builder"
INFO_ROLE = "raphael-verifier"
NATIVE_OPTIONS = {"providers": [
    {"slug": "anthropic", "name": "Anthropic", "authenticated": True, "total_models": 2,
     "models": ["claude-sonnet-5", "claude-opus-5-5"]},
    {"slug": "openrouter", "name": "OpenRouter", "authenticated": True, "total_models": 1,
     "models": ["anthropic/claude-opus-5-5"]},
    {"slug": "openai-codex", "name": "OpenAI Codex", "authenticated": False, "total_models": 2,
     "models": ["gpt-6.1-sol", "gpt-5.6-terra"]},
]}
NATIVE_OAUTH = {"providers": [
    {"id": "anthropic", "name": "Anthropic (Claude)", "flow": "pkce",
     "status": {"logged_in": False, "source": None}},
    {"id": "claude-code", "name": "Claude Code", "flow": "external",
     "status": {"logged_in": True, "source": "claude_code_cli"}},
    {"id": "openai-codex", "name": "OpenAI Codex", "flow": "device_code",
     "status": {"logged_in": False, "source": None}},
    {"id": "nous", "name": "Nous Portal", "flow": "device_code",
     "status": {"logged_in": True, "source": "hermes_auth_store"}},
]}


@pytest.mark.asyncio
async def test_the_models_examples_carry_both_projections_the_info_and_the_refusal(
    codex_accounts, owner_payload_example,
):
    for role in (OPTIONS_ROLE, INFO_ROLE):
        codex_accounts.profile(role)
    machine = _options_request(machine=True)
    revision = profile_routes._profile_route_revision(get_profile_dir(OPTIONS_ROLE))
    live = {
        "model_options": model_policy.project_options_payload(
            NATIVE_OPTIONS, profile=OPTIONS_ROLE, revision=revision),
        "oauth_providers": model_policy.project_oauth_payload(NATIVE_OAUTH),
        "model_info": web_server.get_model_info(machine, profile=INFO_ROLE),
    }
    with pytest.raises(HTTPException) as refused:
        await web_server.get_model_options(machine, profile=INFO_ROLE, include_unconfigured=True)
    response = await http_exception_handler(machine, refused.value)
    live["invalid_model_options"] = {"status": response.status_code, "body": json.loads(response.body)}

    saved = {kind: owner_payload_example(FAMILY, kind, payload) for kind, payload in live.items()}

    # The machine info is the owner's own read of the same role, narrowed.
    full = web_server.get_model_info(_options_request(machine=False), profile=INFO_ROLE)
    assert live["model_info"] == {key: full[key] for key in live["model_info"]}
    admitted = model_policy.admitted_provider_ids()
    native_rows = {raw["slug"]: raw for raw in NATIVE_OPTIONS["providers"]}
    native_logins = {raw["id"]: raw["status"]["logged_in"] for raw in NATIVE_OAUTH["providers"]}
    for examples in (live, saved):
        options = examples["model_options"]
        rows = options["providers"]
        # Only the admitted providers stay, in native order: one assigned, one not.
        assert [row["slug"] for row in rows] == [slug for slug in native_rows if slug in admitted]
        assert {row["assignment"] is None for row in rows} == {True, False}
        for row in rows:
            native = native_rows[row["slug"]]
            assert row["authenticated"] == native["authenticated"]
            assert (row["assignment"] is None) == (row["task_routes"] is None)
            routes = [row["assignment"], *row["task_routes"].values()] if row["assignment"] else []
            assert {(r["profile"], r["provider"]) for r in routes} <= {(options["profile"], row["slug"])}
            # A role is offered exactly the native models one of its routes runs.
            runs = {route["model"] for route in routes}
            assert row["models"] == [model for model in native["models"] if model in runs]
        providers = examples["oauth_providers"]["providers"]
        logins = {row["id"]: row["status"]["logged_in"] for row in providers}
        assert list(logins) == [provider for provider in native_logins if provider in admitted]
        # The one Claude Code login is the owner's Anthropic connection.
        assert logins == {
            provider: native_logins[provider]
            or (provider == "anthropic" and native_logins["claude-code"])
            for provider in logins
        }
        assert set(examples["model_info"]) < set(full)
        refusal = examples["invalid_model_options"]
        assert refusal["status"] >= 400 and "detail" in refusal["body"]


def test_the_models_examples_keep_to_every_closed_vocabulary():
    """The saved Models examples, read together, take every value of each row
    of the table; the fixture holds each one to its row."""
    contract.assert_closed_vocabularies_covered(FAMILY)
