"""Item 32Q: Raphael may manage only its admitted native model lanes."""

from types import SimpleNamespace
from pathlib import Path

import pytest
from fastapi import HTTPException

from hermes_cli import web_server
from hermes_cli.web_models import ProfileModelUpdate
from hermes_cli.web_routers import profiles as profile_routes
from plugins.dashboard_auth.raphael_workspace import model_policy
from plugins.dashboard_auth.raphael_workspace.model_policy import (
    assignment_for,
    model_machine_request,
    project_oauth_poll,
    project_oauth_result,
    project_oauth_start,
    require_models_machine_profile,
    validate_assignment,
)


@pytest.mark.parametrize(
    ("profile", "provider", "family", "effort"),
    [
        ("raphael-planner", "anthropic", "claude-sonnet-", "max"),
        ("default", "anthropic", "claude-opus-", "max"),
        ("raphael-planner", "openai-codex", "gpt-5.6-sol", "high"),
        ("default", "openai-codex", "gpt-5.6-sol", "xhigh"),
    ],
)
def test_admitted_assignment_is_role_bound(profile, provider, family, effort):
    assignment = assignment_for(profile, provider)
    assert assignment.provider == provider
    assert assignment.model.startswith(family)
    assert assignment.reasoning_effort == effort
    validate_assignment(
        profile,
        assignment.provider,
        assignment.model,
        assignment.reasoning_effort,
        disable_fallbacks=True,
    )


@pytest.mark.parametrize(
    ("profile", "provider", "model", "effort", "disable_fallbacks"),
    [
        ("raphael-planner", "anthropic", "claude-fable-5", "max", True),
        ("default", "anthropic", "claude-opus-ultracode-1", "max", True),
        ("raphael-planner", "anthropic", "claude-opus-5", "max", True),
        ("default", "anthropic", "claude-sonnet-5", "max", True),
        ("default", "openai-codex", "gpt-5.6-terra", "xhigh", True),
        ("default", "openai-codex", "gpt-5.6-sol", "max", True),
        ("default", "openai-codex", "gpt-5.6-sol", "xhigh", False),
        ("other-profile", "anthropic", "claude-opus-5", "max", True),
    ],
)
def test_unadmitted_or_fallback_capable_assignment_fails_closed(
    profile, provider, model, effort, disable_fallbacks
):
    with pytest.raises(ValueError):
        validate_assignment(
            profile,
            provider,
            model,
            effort,
            disable_fallbacks=disable_fallbacks,
        )


def test_machine_request_is_bound_to_the_models_provider():
    allowed = SimpleNamespace(
        state=SimpleNamespace(
            token_authenticated=True,
            token_principal=SimpleNamespace(provider="raphael-models-token"),
        )
    )
    interactive = SimpleNamespace(state=SimpleNamespace(token_authenticated=False))
    wrong_machine = SimpleNamespace(
        state=SimpleNamespace(
            token_authenticated=True,
            token_principal=SimpleNamespace(provider="raphael-connections-token"),
        )
    )

    assert model_machine_request(allowed) is True
    assert model_machine_request(interactive) is False
    assert model_machine_request(wrong_machine) is False


def test_profile_model_write_persists_canonical_reasoning_effort(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    saved = []
    monkeypatch.setattr(
        web_server,
        "load_config",
        lambda: {
            "model": {"default": "gpt-5.6-sol", "provider": "openai-codex"},
            "agent": {"reasoning_effort": "high"},
            "fallback_model": "legacy-fallback",
            "fallback_providers": ["legacy-provider"],
        },
    )
    monkeypatch.setattr(web_server, "save_config", saved.append)

    web_server._write_profile_model(
        tmp_path,
        "anthropic",
        "claude-opus-5",
        reasoning_effort=" MAX ",
        disable_fallbacks=True,
    )

    assert saved[0]["agent"]["reasoning_effort"] == "max"
    assert saved[0]["fallback_providers"] == []
    assert "fallback_model" not in saved[0]


def test_machine_profile_scope_is_exact_and_interactive_callers_are_unchanged():
    machine = SimpleNamespace(
        state=SimpleNamespace(
            token_authenticated=True,
            token_principal=SimpleNamespace(provider="raphael-models-token"),
        )
    )
    interactive = SimpleNamespace(state=SimpleNamespace(token_authenticated=False))

    assert (
        require_models_machine_profile(
            machine, "raphael-planner", allowed=(None, "default", "raphael-planner")
        )
        == "raphael-planner"
    )
    assert (
        require_models_machine_profile(machine, "current", allowed=(None, "default"))
        is None
    )
    with pytest.raises(HTTPException) as exc:
        require_models_machine_profile(
            machine, "other-profile", allowed=(None, "default")
        )
    assert exc.value.status_code == 400
    assert (
        require_models_machine_profile(
            interactive, "other-profile", allowed=(None, "default")
        )
        == "other-profile"
    )


def test_machine_oauth_projection_removes_error_and_timing_details():
    assert project_oauth_start(
        "openai-codex",
        {
            "session_id": "session-safe",
            "flow": "device_code",
            "expires_in": 900,
            "user_code": "ABCD-EFGH",
            "verification_url": "https://auth.openai.com/codex/device",
            "poll_interval": 5,
            "internal": "SECRET",
        },
    ) == {
        "session_id": "session-safe",
        "flow": "device_code",
        "expires_in": 900,
        "user_code": "ABCD-EFGH",
        "verification_url": "https://auth.openai.com/codex/device",
        "poll_interval": 5,
    }
    assert project_oauth_result({
        "ok": False,
        "status": "error",
        "message": "SECRET provider response",
    }) == {"ok": False, "status": "error"}
    assert project_oauth_poll({
        "session_id": "session-safe",
        "status": "pending",
        "error_message": "SECRET provider response",
        "expires_at": 123,
    }) == {"session_id": "session-safe", "status": "pending"}


@pytest.mark.asyncio
async def test_profile_model_machine_update_returns_exact_applied_assignment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    writes = []
    audits = []
    monkeypatch.setattr(profile_routes, "_resolve_profile_dir", lambda name: tmp_path)
    monkeypatch.setattr(
        profile_routes,
        "_write_profile_model",
        lambda *args, **kwargs: writes.append((args, kwargs)),
    )
    monkeypatch.setattr(
        model_policy,
        "audit_models_machine_success",
        lambda *args, **kwargs: audits.append((args, kwargs)),
    )
    request = SimpleNamespace(
        method="PUT",
        url=SimpleNamespace(path="/api/profiles/default/model"),
        state=SimpleNamespace(
            token_authenticated=True,
            token_principal=SimpleNamespace(provider="raphael-models-token"),
        ),
    )
    body = ProfileModelUpdate(
        provider="anthropic",
        model="claude-opus-5",
        reasoning_effort="max",
        disable_fallbacks=True,
    )

    result = await profile_routes.update_profile_model_endpoint(
        "default", body, request
    )

    assert result == {
        "ok": True,
        "provider": "anthropic",
        "model": "claude-opus-5",
        "reasoning_effort": "max",
    }
    assert writes
    assert audits
