"""Item 32Q: Raphael may manage only its admitted native model lanes."""

import json
import stat
from types import SimpleNamespace
from pathlib import Path

import pytest
import yaml
from fastapi import HTTPException

from hermes_cli import web_server
from hermes_cli.web_models import (
    ProfileModelBatchEntry,
    ProfileModelBatchUpdate,
    ProfileModelUpdate,
)
from hermes_cli.web_routers import profiles as profile_routes
from plugins.dashboard_auth.raphael_workspace import model_policy
from plugins.dashboard_auth.raphael_workspace.model_policy import (
    assignment_for,
    configured_assignment_for,
    model_machine_request,
    normalize_execution_tier,
    project_oauth_poll,
    project_oauth_result,
    project_oauth_start,
    project_options_payload,
    require_models_machine_profile,
    resolve_task_assignment,
    task_assignment_for,
    validate_assignment,
    validate_runtime_assignment,
)


@pytest.mark.parametrize(
    ("profile", "provider", "family", "effort"),
    [
        ("raphael-planner", "anthropic", "claude-sonnet-", "max"),
        ("default", "anthropic", "claude-opus-", "max"),
        ("raphael-business", "anthropic", "claude-sonnet-", "high"),
        ("raphael-designer", "anthropic", "claude-opus-", "max"),
        ("raphael-claude-worker", "anthropic", "claude-sonnet-", "max"),
        ("raphael-builder", "anthropic", "claude-sonnet-", "max"),
        ("raphael-verifier", "openai-codex", "gpt-5.6-sol", "max"),
        ("raphael-verifier", "anthropic", "claude-opus-", "max"),
        ("raphael-planner", "openai-codex", "gpt-5.6-sol", "max"),
        ("default", "openai-codex", "gpt-5.6-sol", "max"),
        ("raphael-business", "openai-codex", "gpt-5.6-terra", "max"),
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
        ("default", "openai-codex", "gpt-5.6-sol", "xhigh", True),
        ("default", "openai-codex", "gpt-5.6-sol", "xhigh", False),
        ("other-profile", "anthropic", "claude-opus-5", "max", True),
        # High-risk coding may not leave the Claude family, and independent
        # verification may only fall back to the strongest Claude model, never
        # to the builder's own Sonnet lane.
        ("raphael-builder", "openai-codex", "gpt-5.6-terra", "max", True),
        ("raphael-builder", "openai-codex", "gpt-5.6-sol", "max", True),
        ("raphael-verifier", "anthropic", "claude-sonnet-5", "max", True),
        ("raphael-verifier", "anthropic", "claude-opus-5", "high", True),
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


def test_task_route_uses_opus_max_only_for_deep_anthropic_work():
    routine = task_assignment_for(
        "raphael-claude-worker", "anthropic", "routine"
    )
    deep = task_assignment_for("raphael-claude-worker", "anthropic", "deep")

    assert (routine.model, routine.reasoning_effort) == (
        "claude-sonnet-5",
        "max",
    )
    assert (deep.model, deep.reasoning_effort) == ("claude-opus-5", "max")
    assert deep.model_label == "Claude Opus 5 + Claude Code"
    assert task_assignment_for(
        "raphael-verifier", "openai-codex", "deep"
    ) == assignment_for("raphael-verifier", "openai-codex")


@pytest.mark.parametrize("tier", ["routine", "deep"])
def test_no_builder_lane_leaves_the_claude_family(tier):
    """Delivery/infrastructure work is high-risk coding: Claude only.

    Both directions matter. The OpenAI builder assignment must not exist
    (option exposure and setup selection both read the matrix), and no lock
    naming that route may be mintable.
    """
    with pytest.raises(ValueError):
        assignment_for("raphael-builder", "openai-codex")
    with pytest.raises(ValueError):
        task_assignment_for("raphael-builder", "openai-codex", tier)
    with pytest.raises(ValueError):
        model_policy.mint_policy_lock(
            "raphael-builder", "openai-codex", "gpt-5.6-terra", "max", tier,
        )
    # The admitted deep builder lane is Claude Opus 5 at max, and nothing else.
    deep = task_assignment_for("raphael-builder", "anthropic", "deep")
    assert (deep.provider, deep.model, deep.reasoning_effort) == (
        "anthropic", "claude-opus-5", "max",
    )
    assert model_policy.mint_policy_lock(
        "raphael-builder", "anthropic", "claude-opus-5", "max", "deep",
    ).startswith(f"{model_policy.POLICY_LOCK_AUTHORITY}:v")


@pytest.mark.parametrize("tier", ["routine", "deep"])
def test_independent_review_recommends_openai_and_admits_only_the_opus_fallback(tier):
    """The verifier stays independent of the builder's lane.

    The OpenAI route remains the recommended one; the dated Anthropic fallback
    (admitted 2026-09-04) resolves to Claude Opus 5 / max on every tier and is
    never presented as recommended, and the builder's Sonnet lane is refused.
    """
    fallback = assignment_for("raphael-verifier", "anthropic")
    assert (fallback.model, fallback.reasoning_effort, fallback.recommended) == (
        "claude-opus-5", "max", False,
    )
    assert task_assignment_for("raphael-verifier", "anthropic", tier) == fallback
    assert model_policy.mint_policy_lock(
        "raphael-verifier", "anthropic", "claude-opus-5", "max", tier,
    ).startswith(f"{model_policy.POLICY_LOCK_AUTHORITY}:v")
    with pytest.raises(ValueError):
        model_policy.mint_policy_lock(
            "raphael-verifier", "anthropic", "claude-sonnet-5", "max", tier,
        )
    verifier = task_assignment_for("raphael-verifier", "openai-codex", tier)
    assert verifier.recommended is True
    assert (verifier.provider, verifier.model, verifier.reasoning_effort) == (
        "openai-codex", "gpt-5.6-sol", "max",
    )
    assert model_policy.mint_policy_lock(
        "raphael-verifier", "openai-codex", "gpt-5.6-sol", "max", tier,
    ).startswith(f"{model_policy.POLICY_LOCK_AUTHORITY}:v")


def test_removed_lanes_are_not_exposed_as_selectable_options():
    """A role's option payload offers no model it may not actually run."""
    native = {
        "providers": [
            {
                "slug": "anthropic",
                "authenticated": True,
                "models": ["claude-sonnet-5", "claude-opus-5"],
            },
            {
                "slug": "openai-codex",
                "authenticated": True,
                "models": ["gpt-5.6-sol", "gpt-5.6-terra"],
            },
        ]
    }
    builder = project_options_payload(native, profile="raphael-builder")
    by_slug = {row["slug"]: row for row in builder["providers"]}
    assert by_slug["openai-codex"]["assignment"] is None
    assert by_slug["openai-codex"]["task_routes"] is None
    assert by_slug["openai-codex"]["models"] == []
    assert by_slug["anthropic"]["models"] == ["claude-sonnet-5", "claude-opus-5"]

    verifier = project_options_payload(native, profile="raphael-verifier")
    by_slug = {row["slug"]: row for row in verifier["providers"]}
    assert by_slug["anthropic"]["assignment"]["model"] == "claude-opus-5"
    assert by_slug["anthropic"]["assignment"]["recommended"] is False
    assert by_slug["anthropic"]["models"] == ["claude-opus-5"]
    assert by_slug["openai-codex"]["assignment"]["recommended"] is True
    assert by_slug["openai-codex"]["models"] == ["gpt-5.6-sol"]


def test_task_route_rejects_invented_tiers_and_forbidden_runtime_choices():
    with pytest.raises(ValueError):
        normalize_execution_tier("ultra")
    with pytest.raises(ValueError):
        validate_runtime_assignment(
            "raphael-claude-worker",
            "anthropic",
            "claude-fable-5",
            "max",
            disable_fallbacks=True,
        )
    with pytest.raises(ValueError):
        validate_runtime_assignment(
            "raphael-claude-worker",
            "anthropic",
            "claude-opus-5",
            "max",
            disable_fallbacks=False,
        )


def test_model_options_project_one_canonical_base_and_task_routes():
    result = project_options_payload(
        {
            "providers": [
                {
                    "slug": "anthropic",
                    "authenticated": True,
                    "models": [
                        "claude-sonnet-5",
                        "claude-opus-5",
                        "claude-fable-5",
                        "claude-opus-ultracode-1",
                    ],
                },
                {
                    "slug": "openai-codex",
                    "authenticated": True,
                    "models": ["gpt-5.6-sol", "gpt-5.6-terra"],
                },
            ]
        },
        profile="raphael-claude-worker",
    )

    assert result["profile"] == "raphael-claude-worker"
    anthropic = result["providers"][0]
    assert anthropic["models"] == ["claude-sonnet-5", "claude-opus-5"]
    assert anthropic["assignment"]["model"] == "claude-sonnet-5"
    # Workspace consumes this as a closed set of exactly the two task classes,
    # each carrying the same fields as the base assignment.
    assert set(anthropic["task_routes"]) == {"routine", "deep"}
    assert anthropic["task_routes"]["routine"]["model"] == "claude-sonnet-5"
    assert anthropic["task_routes"]["deep"]["model"] == "claude-opus-5"
    for route in anthropic["task_routes"].values():
        assert set(route) == set(anthropic["assignment"])
        assert (route["profile"], route["provider"]) == (
            "raphael-claude-worker",
            "anthropic",
        )
    # A role/provider pair with no admitted assignment publishes no routes,
    # which is how Workspace knows that provider cannot serve the role.
    codex = result["providers"][1]
    assert (codex["assignment"], codex["task_routes"]) == (None, None)
    assert "fable" not in str(result).lower()
    assert "ultracode" not in str(result).lower()


def test_new_work_uses_the_provider_currently_selected_for_its_role(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setattr(
        "hermes_cli.profiles.get_profile_dir", lambda _profile: tmp_path
    )
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        lambda: {
            "model": {
                "provider": "openai-codex",
                "default": "gpt-5.6-sol",
            },
            "agent": {"reasoning_effort": "max"},
            "fallback_providers": [],
        },
    )
    monkeypatch.setattr(
        "hermes_constants.set_hermes_home_override", lambda _path: object()
    )
    monkeypatch.setattr(
        "hermes_constants.reset_hermes_home_override", lambda _token: None
    )

    configured = configured_assignment_for("raphael-planner")
    deep = resolve_task_assignment("raphael-planner", "deep")

    assert configured.provider == "openai-codex"
    assert (deep.provider, deep.model, deep.reasoning_effort) == (
        "openai-codex",
        "gpt-5.6-sol",
        "max",
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


def test_profile_model_write_is_a_compare_and_swap_when_a_revision_is_stated(
    tmp_path: Path,
):
    """Stating a revision must still reach save_config's compare-and-swap.

    Nothing is monkeypatched: this is the real config write against a real
    profile home, so the revision a caller echoes back is the one the CAS
    compares against.
    """
    from hermes_cli import config as hermes_config

    first = web_server._write_profile_model(
        tmp_path, "anthropic", "claude-opus-5",
        reasoning_effort="max", disable_fallbacks=True,
    )
    assert first == profile_routes._profile_route_revision(tmp_path)

    # The revision the caller holds is accepted...
    second = web_server._write_profile_model(
        tmp_path, "openai-codex", "gpt-5.6-sol",
        reasoning_effort="max", disable_fallbacks=True,
        expected_revision=first,
    )
    assert second == profile_routes._profile_route_revision(tmp_path) != first

    # ...and the now-stale one is refused instead of silently overwriting.
    with pytest.raises(hermes_config.RouteRevisionConflict):
        web_server._write_profile_model(
            tmp_path, "anthropic", "claude-opus-5",
            reasoning_effort="max", disable_fallbacks=True,
            expected_revision=first,
        )
    assert profile_routes._profile_route_revision(tmp_path) == second


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


def _machine_request():
    return SimpleNamespace(
        method="PUT",
        url=SimpleNamespace(path="/api/profiles/default/model"),
        state=SimpleNamespace(
            token_authenticated=True,
            token_principal=SimpleNamespace(provider="raphael-models-token"),
        ),
    )


@pytest.fixture
def machine_route_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """A real profile home plus a real, isolated enrollment registry."""
    home = tmp_path / "hermes"
    home.mkdir()
    registry = tmp_path / "raphael" / "model_policy_enrollment.json"
    monkeypatch.setattr(model_policy, "enrollment_path", lambda: registry)
    monkeypatch.setattr(profile_routes, "_resolve_profile_dir", lambda name: home)
    monkeypatch.setattr(
        "hermes_cli.profiles.get_profile_dir", lambda _profile: home
    )
    audits: list = []
    monkeypatch.setattr(
        model_policy,
        "journal_models_machine_batch_success",
        lambda *args, **kwargs: (
            audits.append((args, kwargs)) or SimpleNamespace(commit=lambda: None)
        ),
    )
    return SimpleNamespace(home=home, registry=registry, audits=audits)


@pytest.mark.asyncio
async def test_profile_model_machine_update_returns_exact_applied_assignment(
    machine_route_env,
):
    body = ProfileModelUpdate(
        provider="anthropic",
        model="claude-opus-5",
        reasoning_effort="max",
        disable_fallbacks=True,
    )

    result = await profile_routes.update_profile_model_endpoint(
        "default", body, _machine_request()
    )

    assert result == {
        "ok": True,
        "provider": "anthropic",
        "model": "claude-opus-5",
        "reasoning_effort": "max",
        # The Hermes-owned route revision the caller must echo back on its
        # next conditional write.
        "revision": profile_routes._profile_route_revision(machine_route_env.home),
    }
    assert machine_route_env.audits
    # The write is not complete until the role is durably governed.
    assert model_policy.is_profile_enrolled("default") is True


@pytest.mark.asyncio
@pytest.mark.parametrize("raw_body", [
    {"provider": "anthropic", "model": "claude-opus-5", "reasoning_effort": "max",
     "disable_fallbacks": True, "expected_revision": None},
])
async def test_present_but_null_expected_revision_is_refused_before_any_write(
    machine_route_env, raw_body
):
    """Pydantic collapses omitted and explicit null; the endpoint must not.

    An explicit null would otherwise silently downgrade a conditional write to
    an unconditional one — the exact thing the caller was protecting against.
    """
    body = ProfileModelUpdate(**raw_body)
    assert body.expected_revision is None

    with pytest.raises(HTTPException) as exc:
        await profile_routes.update_profile_model_endpoint(
            "default", body, _machine_request()
        )

    assert exc.value.status_code == 400
    assert not (machine_route_env.home / "config.yaml").exists()
    assert not machine_route_env.registry.exists()


@pytest.mark.asyncio
async def test_omitted_expected_revision_stays_supported(machine_route_env):
    """Documented compatibility: not sending the field is still allowed."""
    body = ProfileModelUpdate(
        provider="anthropic", model="claude-opus-5",
        reasoning_effort="max", disable_fallbacks=True,
    )
    assert "expected_revision" not in body.model_fields_set

    result = await profile_routes.update_profile_model_endpoint(
        "default", body, _machine_request()
    )
    assert result["ok"] is True


@pytest.mark.asyncio
async def test_stale_and_exact_expected_revision_behave_as_a_compare_and_swap(
    machine_route_env,
):
    first = await profile_routes.update_profile_model_endpoint(
        "default",
        ProfileModelUpdate(
            provider="anthropic", model="claude-opus-5",
            reasoning_effort="max", disable_fallbacks=True,
        ),
        _machine_request(),
    )
    # Exact revision is accepted.
    second = await profile_routes.update_profile_model_endpoint(
        "default",
        ProfileModelUpdate(
            provider="openai-codex", model="gpt-5.6-sol",
            reasoning_effort="max", disable_fallbacks=True,
            expected_revision=first["revision"],
        ),
        _machine_request(),
    )
    assert second["revision"] != first["revision"]
    # The now-stale one is refused.
    with pytest.raises(HTTPException) as exc:
        await profile_routes.update_profile_model_endpoint(
            "default",
            ProfileModelUpdate(
                provider="anthropic", model="claude-opus-5",
                reasoning_effort="max", disable_fallbacks=True,
                expected_revision=first["revision"],
            ),
            _machine_request(),
        )
    assert exc.value.status_code == 409
    assert (
        profile_routes._profile_route_revision(machine_route_env.home)
        == second["revision"]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_point", ["enrollment", "audit"])
async def test_a_post_write_failure_restores_config_and_enrollment(
    machine_route_env, monkeypatch: pytest.MonkeyPatch, failure_point: str
):
    """A failure response must never leave the new route live.

    Config, enrollment and the strict audit record are one operation: if the
    later steps fail, the exact prior config bytes and the exact prior
    enrollment registry are restored before the error is returned.
    """
    home = machine_route_env.home
    # Establish a first, successful state to roll back TO.
    first = await profile_routes.update_profile_model_endpoint(
        "default",
        ProfileModelUpdate(
            provider="anthropic", model="claude-opus-5",
            reasoning_effort="max", disable_fallbacks=True,
        ),
        _machine_request(),
    )
    config_before = (home / "config.yaml").read_text(encoding="utf-8")
    registry_before = machine_route_env.registry.read_text(encoding="utf-8")

    def _boom(*_args, **_kwargs):
        raise RuntimeError("injected")

    if failure_point == "enrollment":
        monkeypatch.setattr(model_policy, "enroll_profiles", _boom)
    else:
        monkeypatch.setattr(
            model_policy, "journal_models_machine_batch_success", _boom
        )

    with pytest.raises(HTTPException):
        await profile_routes.update_profile_model_endpoint(
            "default",
            ProfileModelUpdate(
                provider="openai-codex", model="gpt-5.6-sol",
                reasoning_effort="max", disable_fallbacks=True,
                expected_revision=first["revision"],
            ),
            _machine_request(),
        )

    # Re-read from disk, not from memory: the route that is actually live.
    assert (home / "config.yaml").read_text(encoding="utf-8") == config_before
    assert machine_route_env.registry.read_text(encoding="utf-8") == registry_before
    assert profile_routes._profile_route_revision(home) == first["revision"]


def _enroll_worker(
    registry_path: str, profile: str, barrier_dir: str, index: int, workers: int
):
    """Enroll one profile, released together with every sibling process."""
    import os
    import time as _time
    from pathlib import Path as _Path

    from plugins.dashboard_auth.raphael_workspace import model_policy as mp

    mp.enrollment_path = lambda: _Path(registry_path)
    (_Path(barrier_dir) / f"ready-{index}").write_text("1", encoding="utf-8")
    deadline = _time.monotonic() + 30
    while len(os.listdir(barrier_dir)) < workers:
        if _time.monotonic() >= deadline:
            raise TimeoutError("barrier never released")
        _time.sleep(0.005)
    mp.enroll_profile(profile)


def test_concurrent_enrollments_of_different_profiles_all_survive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Enrollment is a shared read-modify-replace; no role may be lost.

    Barrier-driven so every process reads the registry at the same instant and
    then writes it back. Without one cross-process lock the later writer drops
    the earlier one's role. No retry: one attempt each must be enough.
    """
    import multiprocessing as mp_lib

    registry = tmp_path / "raphael" / "model_policy_enrollment.json"
    barrier_dir = tmp_path / "barrier"
    barrier_dir.mkdir()
    roles = [
        "default", "raphael-planner", "raphael-business",
        "raphael-designer", "raphael-builder", "raphael-verifier",
    ]
    ctx = mp_lib.get_context("spawn")
    procs = [
        ctx.Process(
            target=_enroll_worker,
            args=(str(registry), role, str(barrier_dir), index, len(roles)),
        )
        for index, role in enumerate(roles)
    ]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join(timeout=90)
    assert [proc.exitcode for proc in procs] == [0] * len(roles)

    monkeypatch.setattr(model_policy, "enrollment_path", lambda: registry)
    assert model_policy.enrolled_profile_ids() == frozenset(roles)


@pytest.fixture
def machine_batch_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Real per-profile homes plus a real, isolated enrollment registry."""
    root = tmp_path / "profiles"
    root.mkdir()
    registry = tmp_path / "raphael" / "model_policy_enrollment.json"

    def _home(name: str) -> Path:
        directory = root / str(name)
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    monkeypatch.setattr(model_policy, "enrollment_path", lambda: registry)
    monkeypatch.setattr(profile_routes, "_resolve_profile_dir", _home)
    monkeypatch.setattr("hermes_cli.profiles.get_profile_dir", _home)
    audited: list = []
    monkeypatch.setattr(
        model_policy,
        "journal_models_machine_batch_success",
        lambda *args, **kwargs: (
            audited.extend(profile for profile, _provider in kwargs["roles"])
            or SimpleNamespace(commit=lambda: None)
        ),
    )
    return SimpleNamespace(home=_home, registry=registry, audited=audited)


def _machine_batch_request():
    return SimpleNamespace(
        method="POST",
        url=SimpleNamespace(path="/api/profiles/model-batch"),
        state=SimpleNamespace(
            token_authenticated=True,
            token_principal=SimpleNamespace(provider="raphael-models-token"),
        ),
    )


def _batch_entry(profile: str, provider: str, **overrides) -> ProfileModelBatchEntry:
    """One entry on the role's admitted route for that provider."""
    admitted = assignment_for(profile, provider)
    return ProfileModelBatchEntry(**{
        "profile": profile,
        "provider": admitted.provider,
        "model": admitted.model,
        "reasoning_effort": admitted.reasoning_effort,
        **overrides,
    })


async def _apply_batch(entries) -> dict:
    return await profile_routes.update_profile_model_batch_endpoint(
        ProfileModelBatchUpdate(assignments=list(entries)), _machine_batch_request(),
    )


@pytest.mark.asyncio
async def test_batch_governs_every_role_including_one_that_already_matches(
    machine_batch_env,
):
    """Matching config bytes are not the same fact as being governed.

    A role skipped for "already correct" would stay selected-looking and
    unenrolled, so the operation covers it too — and because the write is a
    no-op on the effective route, covering it does not manufacture a change.
    """
    home = machine_batch_env.home
    roles = ["default", "raphael-planner", "raphael-business"]
    # ``default`` is put on exactly the route the batch will ask for, outside
    # the policy, so its bytes already match while nothing about it is enrolled.
    matching = assignment_for("default", "anthropic")
    home("default").joinpath("config.yaml").write_text(
        yaml.safe_dump({
            "model": {"provider": matching.provider, "default": matching.model},
            "agent": {"reasoning_effort": matching.reasoning_effort},
            "fallback_providers": [],
        }),
        encoding="utf-8",
    )
    assert model_policy.enrolled_profile_ids() == frozenset()
    before = profile_routes._profile_route_revision(home("default"))

    result = await _apply_batch(_batch_entry(role, "anthropic") for role in roles)

    assert result["ok"] is True
    assert {row["profile"] for row in result["applied"]} == set(roles)
    # Validated, enrolled and audited — every role, no exception for the one
    # whose file did not need to change.
    assert model_policy.enrolled_profile_ids() == frozenset(roles)
    assert sorted(machine_batch_env.audited) == sorted(roles)
    # ...and the already-correct role's effective route never moved.
    after = profile_routes._profile_route_revision(home("default"))
    assert after == before
    assert next(
        row["revision"] for row in result["applied"] if row["profile"] == "default"
    ) == before


def test_an_unprovable_audit_rollback_never_fails_the_batch(
    monkeypatch: pytest.MonkeyPatch,
):
    """The commit line reached the log and could not be proven gone, so the log
    may already assert this batch. Failing here made the caller revert the
    routes, which is the exact split the two-phase record exists to prevent."""
    from hermes_cli.dashboard_auth.audit import AuditBatch, AuditRollbackUncertain

    def _uncertain(_batch):
        raise AuditRollbackUncertain("the commit line could not be removed")

    monkeypatch.setattr(model_policy, "commit_audit_batch", _uncertain)
    audit = model_policy._JournalledBatchAudit(
        AuditBatch("abc123", ({"event": "token_auth_success"},))
    )
    # Rolled FORWARD: no exception, so nothing reverts the applied routes.
    audit.commit()


def test_an_audit_write_that_left_nothing_behind_still_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
):
    """The adjacent case: the log says nothing about this operation, so the
    caller's rollback makes the log and the world agree."""
    from hermes_cli.dashboard_auth.audit import AuditBatch, AuditWriteError

    def _failed(_batch):
        raise AuditWriteError("nothing was written")

    monkeypatch.setattr(model_policy, "commit_audit_batch", _failed)
    audit = model_policy._JournalledBatchAudit(
        AuditBatch("abc123", ({"event": "token_auth_success"},))
    )
    with pytest.raises(HTTPException) as exc:
        audit.commit()
    assert exc.value.status_code == 503


@pytest.mark.asyncio
async def test_a_batch_whose_audit_rollback_is_unprovable_keeps_its_routes(
    machine_batch_env, monkeypatch: pytest.MonkeyPatch
):
    """End to end: a committed audit may never coexist with reverted routes."""
    from hermes_cli.dashboard_auth.audit import AuditBatch, AuditRollbackUncertain

    home = machine_batch_env.home
    roles = ["default", "raphael-planner"]

    def _uncertain(_batch):
        raise AuditRollbackUncertain("the commit line could not be removed")

    monkeypatch.setattr(model_policy, "commit_audit_batch", _uncertain)
    monkeypatch.setattr(
        model_policy,
        "journal_models_machine_batch_success",
        lambda *args, **kwargs: model_policy._JournalledBatchAudit(
            AuditBatch("abc123", ({"event": "token_auth_success"},))
        ),
    )

    result = await _apply_batch(_batch_entry(role, "anthropic") for role in roles)

    assert result["ok"] is True
    # The routes and the enrollment stand; nothing was rolled back.
    assert model_policy.enrolled_profile_ids() == frozenset(roles)
    for role in roles:
        expected = assignment_for(role, "anthropic")
        live = yaml.safe_load(
            home(role).joinpath("config.yaml").read_text(encoding="utf-8")
        )
        assert live["model"]["default"] == expected.model


@pytest.mark.asyncio
@pytest.mark.parametrize("index", [0, 1, 2])
async def test_a_present_null_expected_revision_on_any_batch_item_is_refused(
    machine_batch_env, index
):
    """The single-write rule holds for EVERY item, not just the first.

    Pydantic collapses omitted and explicit null, so a null anywhere in the
    batch would silently downgrade that role's conditional write.
    """
    roles = ["default", "raphael-planner", "raphael-business"]
    entries = [_batch_entry(role, "anthropic") for role in roles]
    entries[index] = _batch_entry(
        roles[index], "anthropic", expected_revision=None,
    )
    assert "expected_revision" in entries[index].model_fields_set
    assert all(
        "expected_revision" not in entry.model_fields_set
        for position, entry in enumerate(entries)
        if position != index
    )

    with pytest.raises(HTTPException) as exc:
        await _apply_batch(entries)

    assert exc.value.status_code == 400
    # Refused in the preflight: no role was written and none was enrolled.
    for role in roles:
        assert not machine_batch_env.home(role).joinpath("config.yaml").exists()
    assert not machine_batch_env.registry.exists()


@pytest.mark.asyncio
async def test_batch_rollback_covers_the_profile_whose_own_write_then_failed(
    machine_batch_env, monkeypatch: pytest.MonkeyPatch
):
    """A write-then-raise already changed that profile's file: restore it too.

    Also covers the two snapshot shapes: a role that HAD a config goes back to
    its exact bytes and mode, and a role that had none goes back to having
    none.
    """
    home = machine_batch_env.home
    established = ["default", "raphael-planner"]
    # Deterministic order is sorted, so this role is reached last — every other
    # role has already been written when it raises.
    failing = "raphael-planner"
    # ``raphael-designer`` deliberately has no config at all beforehand.
    fresh = "raphael-designer"

    first = await _apply_batch(
        _batch_entry(role, "anthropic") for role in established
    )
    revisions = {row["profile"]: row["revision"] for row in first["applied"]}
    # A managed deployment leaves config.yaml group-readable, so give one role
    # that mode: its prior mode is then deliberately NOT the 0600 a guarded
    # write leaves behind, and restoring it proves the mode is really restored
    # instead of the post-write mode being compared to itself.
    home("default").joinpath("config.yaml").chmod(0o640)
    before = {
        role: home(role).joinpath("config.yaml").read_text(encoding="utf-8")
        for role in established
    }
    modes = {
        role: stat.S_IMODE(home(role).joinpath("config.yaml").stat().st_mode)
        for role in established
    }
    assert modes["default"] == 0o640
    assert modes[failing] != modes["default"]
    registry_before = machine_batch_env.registry.read_text(encoding="utf-8")
    assert not home(fresh).joinpath("config.yaml").exists()

    real_write = profile_routes._write_profile_model
    write_order: list = []

    def write_then_raise(profile_dir, *args, **kwargs):
        revision = real_write(profile_dir, *args, **kwargs)
        write_order.append(profile_dir.name)
        if profile_dir.name == failing:
            raise RuntimeError("injected after the bytes were already written")
        return revision

    monkeypatch.setattr(profile_routes, "_write_profile_model", write_then_raise)

    with pytest.raises(HTTPException) as exc:
        await _apply_batch([
            *(
                _batch_entry(
                    role, "openai-codex", expected_revision=revisions[role],
                )
                for role in established
            ),
            _batch_entry(fresh, "anthropic"),
        ])
    assert exc.value.status_code == 409
    # Each role's real write returned before the next one started, and the
    # failing role raised only after its own bytes were on disk — so the
    # rollback below covers a preceding role that HAD a config, a preceding
    # role that had none, and the role whose own write then failed.
    assert write_order == ["default", fresh, failing]

    for role in established:
        path = home(role).joinpath("config.yaml")
        assert path.read_text(encoding="utf-8") == before[role]
        assert stat.S_IMODE(path.stat().st_mode) == modes[role]
    assert not home(fresh).joinpath("config.yaml").exists()
    assert machine_batch_env.registry.read_text(encoding="utf-8") == registry_before
    assert model_policy.enrolled_profile_ids() == frozenset(established)


def test_enrollment_restore_puts_back_the_exact_prior_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Rollback must restore content, and absence, exactly."""
    registry = tmp_path / "raphael" / "model_policy_enrollment.json"
    monkeypatch.setattr(model_policy, "enrollment_path", lambda: registry)

    absent_snapshot = model_policy.enroll_profiles(["default"])
    assert absent_snapshot is None
    assert model_policy.is_profile_enrolled("default") is True

    existing_snapshot = model_policy.enroll_profiles(["raphael-planner"])
    assert existing_snapshot is not None
    assert model_policy.is_profile_enrolled("raphael-planner") is True

    model_policy.restore_enrollment(existing_snapshot)
    assert registry.read_text(encoding="utf-8") == existing_snapshot
    assert model_policy.enrolled_profile_ids() == frozenset({"default"})

    model_policy.restore_enrollment(absent_snapshot)
    assert not registry.exists()
    assert model_policy.enrolled_profile_ids() == frozenset()


# ---------------------------------------------------------------------------
# Concurrent enrollment + rollback, and audit truth afterwards
# ---------------------------------------------------------------------------
#
# These use the REAL audit sink (``$HERMES_HOME/logs/dashboard-auth.log``),
# never a list stub: the whole claim under test is what the durable trail says
# after a rolled-back operation.


_AUDITED_ACTION = "assign"


def _audit_log_path():
    from hermes_cli.dashboard_auth.audit import _resolve_log_path

    return _resolve_log_path()


def _audit_lines(path=None) -> list:
    """Everything the log actually asserts happened.

    Reads through the module's own reader, so an entry belonging to a batch
    that was never committed is not counted as a record — which is the whole
    claim these tests make about a rolled-back operation.

    ``path`` must be supplied by any test that undoes its patches first:
    ``monkeypatch.undo()`` also reverts conftest's ``HERMES_HOME`` sandbox, so
    a path-less read after it answers from a different, empty log.
    """
    from hermes_cli.dashboard_auth.audit import read_audit_records

    return read_audit_records(path)


def _raw_audit_lines(path) -> list:
    """What an ordinary one-JSON-object-per-line consumer of this log sees.

    The log's documented production format is one JSON object per line, so
    this is a real reading of it — and it must not report a rolled-back
    operation's roles as allowed either.
    """
    if not path.is_file():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, dict):
            continue
        nested = record.get("records")
        records.extend(
            item for item in (nested if isinstance(nested, list) else [record])
            if isinstance(item, dict)
        )
    return records


def _allowed_roles(entries, action: str = _AUDITED_ACTION) -> list:
    return [
        entry["profile"]
        for entry in entries
        if entry.get("action") == action and entry.get("decision") == "allow"
    ]


def _audited_roles(action: str = _AUDITED_ACTION, path=None) -> list:
    return _allowed_roles(_audit_lines(path), action)


def _real_audit_request():
    """A Models-machine request the REAL strict audit writer can serialise."""
    return SimpleNamespace(
        method="POST",
        url=SimpleNamespace(path="/api/profiles/model-batch"),
        client=SimpleNamespace(host="127.0.0.1"),
        headers={},
        state=SimpleNamespace(
            token_authenticated=True,
            token_route_template="/api/profiles/model-batch",
            token_principal=SimpleNamespace(
                provider="raphael-models-token",
                principal="raphael-models",
                credential_id="cred-1",
            ),
        ),
    )


@pytest.fixture
def real_audit_batch_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Real per-profile homes, a real registry, and the REAL audit sink."""
    root = tmp_path / "profiles"
    root.mkdir()
    registry = tmp_path / "raphael" / "model_policy_enrollment.json"

    def _home(name: str) -> Path:
        directory = root / str(name)
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    monkeypatch.setattr(model_policy, "enrollment_path", lambda: registry)
    monkeypatch.setattr(profile_routes, "_resolve_profile_dir", _home)
    monkeypatch.setattr("hermes_cli.profiles.get_profile_dir", _home)
    return SimpleNamespace(home=_home, registry=registry)


async def _apply_real_batch(entries) -> dict:
    return await profile_routes.update_profile_model_batch_endpoint(
        ProfileModelBatchUpdate(assignments=list(entries)), _real_audit_request(),
    )


@pytest.mark.asyncio
async def test_a_successful_batch_is_audited_once_per_role_in_the_real_log(
    real_audit_batch_env,
):
    roles = ["default", "raphael-business", "raphael-planner"]

    result = await _apply_real_batch(
        _batch_entry(role, "anthropic") for role in roles
    )

    assert result["ok"] is True
    assert sorted(_audited_roles()) == sorted(roles)
    assert model_policy.enrolled_profile_ids() == frozenset(roles)


@pytest.mark.asyncio
async def test_a_rolled_back_batch_leaves_no_effective_success_in_the_real_log(
    real_audit_batch_env, monkeypatch: pytest.MonkeyPatch
):
    """A later role fails after earlier roles were already written.

    Auditing each role as it completed would leave those earlier roles on
    record as ``decision="allow"`` for a change that was reverted. The real
    log must contain nothing for ANY role of the batch.
    """
    home = real_audit_batch_env.home
    roles = ["default", "raphael-business", "raphael-planner"]
    # Deterministic order is sorted, so this one is written last.
    failing = sorted(roles)[-1]

    real_write = profile_routes._write_profile_model
    written: list = []

    def write_then_fail(profile_dir, *args, **kwargs):
        revision = real_write(profile_dir, *args, **kwargs)
        written.append(profile_dir.name)
        if profile_dir.name == failing:
            raise RuntimeError("this role's provider rejected the assignment")
        return revision

    monkeypatch.setattr(profile_routes, "_write_profile_model", write_then_fail)

    with pytest.raises(HTTPException) as exc:
        await _apply_real_batch(_batch_entry(role, "anthropic") for role in roles)

    assert exc.value.status_code == 409
    # The earlier roles really were written before the failure...
    assert written == sorted(roles)
    # ...and none of them is recorded as an allowed change.
    assert _audited_roles() == []
    # Nothing effective remains anywhere else either.
    assert model_policy.enrolled_profile_ids() == frozenset()
    for role in roles:
        assert not home(role).joinpath("config.yaml").exists()


@pytest.mark.asyncio
async def test_an_unwritable_audit_log_refuses_the_batch_and_reverts_it(
    real_audit_batch_env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A change that cannot be durably audited must not stay live."""
    from hermes_cli.dashboard_auth import audit as audit_mod

    roles = ["default", "raphael-business"]
    # The audit log's own parent is a regular FILE, so the directory it needs
    # can never be created — an unwritable sink, with nothing else patched.
    blocked = tmp_path / "audit-parent-is-a-file"
    blocked.write_text("not a directory", encoding="utf-8")
    monkeypatch.setattr(
        audit_mod, "_resolve_log_path", lambda: blocked / "dashboard-auth.log",
    )

    with pytest.raises(HTTPException) as exc:
        await _apply_real_batch(_batch_entry(role, "anthropic") for role in roles)

    assert exc.value.status_code == 503
    assert model_policy.enrolled_profile_ids() == frozenset()
    for role in roles:
        assert not real_audit_batch_env.home(role).joinpath("config.yaml").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["short_write", "fsync"])
async def test_a_partly_durable_audit_leaves_no_effective_allow_record(
    real_audit_batch_env, monkeypatch: pytest.MonkeyPatch, failure
):
    """An append is not all-or-nothing; the record it makes must be.

    ``short_write`` leaves complete early entries on disk. ``fsync`` fails only
    after the WHOLE batch reached the file. Either way the operation is
    reverted, so neither may leave a role recorded as allowed.
    """
    from hermes_cli.dashboard_auth import audit as audit_mod

    roles = ["default", "raphael-business"]
    audit_path = _audit_log_path()
    log_path = audit_mod._resolve_log_path()

    def _is_audit_log(fd) -> bool:
        """Whether this descriptor is the AUDIT log's own.

        ``os.write``/``os.fsync`` are module-wide, and this operation also
        journals its own undo record durably. Narrowing to the audit log is
        what keeps this test about the audit append rather than about whichever
        durable write happens to come first.
        """
        try:
            return (
                log_path.is_file()
                and audit_mod.os.fstat(fd).st_ino == log_path.stat().st_ino
            )
        except (FileNotFoundError, OSError):
            return False

    if failure == "short_write":
        real_write = audit_mod.os.write

        def _short(fd, data):
            if _is_audit_log(fd):
                return real_write(fd, data[: data.index(b"\n") + 1])
            return real_write(fd, data)

        monkeypatch.setattr(audit_mod.os, "write", _short)
    else:
        # Only the AUDIT log's own fsync fails: the config writes have their
        # own, and failing those would be a different (write-time) failure.
        real_fsync = audit_mod.os.fsync

        def _boom(fd):
            if _is_audit_log(fd):
                raise OSError("fsync failed")
            return real_fsync(fd)

        monkeypatch.setattr(audit_mod.os, "fsync", _boom)

    with pytest.raises(HTTPException) as exc:
        await _apply_real_batch(_batch_entry(role, "anthropic") for role in roles)
    monkeypatch.undo()

    assert exc.value.status_code == 503
    assert _audited_roles(path=audit_path) == []
    # And the same is true of the log's own documented format: a reverted
    # operation leaves no line that reads as an allowed role.
    assert _allowed_roles(_raw_audit_lines(audit_path)) == []
    assert model_policy.enrolled_profile_ids() == frozenset()
    for role in roles:
        assert not real_audit_batch_env.home(role).joinpath("config.yaml").exists()


@pytest.mark.asyncio
async def test_a_batch_that_is_never_committed_is_reverted(
    real_audit_batch_env, monkeypatch: pytest.MonkeyPatch
):
    """The commit is the last durable act; losing it reverts the batch."""
    from hermes_cli.dashboard_auth import audit as audit_mod

    roles = ["default", "raphael-business"]
    audit_path = _audit_log_path()

    def _boom(_batch):
        raise audit_mod.AuditWriteError("the batch could not be made durable")

    monkeypatch.setattr(model_policy, "commit_audit_batch", _boom)

    with pytest.raises(HTTPException) as exc:
        await _apply_real_batch(_batch_entry(role, "anthropic") for role in roles)

    assert exc.value.status_code == 503
    assert _audited_roles() == []
    assert _allowed_roles(_raw_audit_lines(audit_path)) == []
    # The prepared bytes really are on disk — and still assert nothing, to
    # either reader.
    assert audit_path.read_text(encoding="utf-8").count(
        audit_mod.AuditEvent.BATCH_PREPARED.value
    ) == len(roles)
    assert model_policy.enrolled_profile_ids() == frozenset()
    for role in roles:
        assert not real_audit_batch_env.home(role).joinpath("config.yaml").exists()


@pytest.mark.asyncio
async def test_a_successful_batch_is_one_allow_record_per_role_to_both_readers(
    real_audit_batch_env,
):
    """The success path is unchanged, and readable straight off the line."""
    roles = ["default", "raphael-business"]
    audit_path = _audit_log_path()

    await _apply_real_batch(_batch_entry(role, "anthropic") for role in roles)

    assert _audited_roles(path=audit_path) == roles
    assert _allowed_roles(_raw_audit_lines(audit_path)) == roles
    assert model_policy.enrolled_profile_ids() == frozenset(roles)


def test_a_batch_audit_record_is_all_or_nothing(tmp_path: Path, monkeypatch):
    """One unencodable entry means NO line is appended, not a partial record."""
    from hermes_cli.dashboard_auth import audit as audit_mod

    log_path = tmp_path / "logs" / "dashboard-auth.log"
    monkeypatch.setattr(audit_mod, "_resolve_log_path", lambda: log_path)

    audit_mod.audit_log_batch(
        [(audit_mod.AuditEvent.TOKEN_AUTH_SUCCESS, {"profile": "default"})],
        strict=True,
    )
    assert len(log_path.read_text(encoding="utf-8").splitlines()) == 1

    with pytest.raises(TypeError):
        audit_mod.audit_log_batch(
            [
                (audit_mod.AuditEvent.TOKEN_AUTH_SUCCESS, {"profile": "first"}),
                (audit_mod.AuditEvent.TOKEN_AUTH_SUCCESS, {"profile": object()}),
            ],
            strict=True,
        )

    # The first entry of the failed batch never landed.
    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert "first" not in log_path.read_text(encoding="utf-8")


def _concurrent_batch_worker(
    role: str, barrier_dir: str, index: int, workers: int, fail: bool, out: str
):
    """Run ONE whole Models-machine batch in its own process.

    Every worker writes its readiness marker and then waits for every sibling,
    so all of them enter the shared enrollment registry at the same instant.
    The failing worker enrolls first and only then raises, from inside the
    registry transaction — the interleaving where a snapshot-based rollback
    could erase a sibling's committed enrollment.
    """
    import asyncio
    import json as _json
    import os as _os
    import time as _time
    from pathlib import Path as _Path

    from hermes_cli.web_models import ProfileModelBatchEntry, ProfileModelBatchUpdate
    from hermes_cli.web_routers import profiles as routes
    from plugins.dashboard_auth.raphael_workspace import model_policy as mp

    if fail:
        def _enrolled_then_boom(request, *, action, roles):
            # Held INSIDE the registry transaction, so the sibling genuinely
            # contends for the lock instead of finishing first by luck.
            _time.sleep(0.25)
            raise RuntimeError("injected after this batch had already enrolled")

        mp.journal_models_machine_batch_success = _enrolled_then_boom

    admitted = mp.assignment_for(role, "anthropic")
    entry = ProfileModelBatchEntry(
        profile=role,
        provider=admitted.provider,
        model=admitted.model,
        reasoning_effort=admitted.reasoning_effort,
    )

    (_Path(barrier_dir) / f"ready-{index}").write_text("1", encoding="utf-8")
    deadline = _time.monotonic() + 30
    while len(_os.listdir(barrier_dir)) < workers:
        if _time.monotonic() >= deadline:
            raise TimeoutError("barrier never released")
        _time.sleep(0.005)

    outcome = {"role": role, "ok": None, "status": None}
    try:
        asyncio.run(
            routes.update_profile_model_batch_endpoint(
                ProfileModelBatchUpdate(assignments=[entry]),
                _real_audit_request(),
            )
        )
        outcome["ok"] = True
    except HTTPException as exc:
        outcome.update(ok=False, status=exc.status_code)
    _Path(out).write_text(_json.dumps(outcome), encoding="utf-8")


def test_a_concurrent_batch_rollback_never_erases_the_other_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Two real processes, one barrier, no retries, no deadlock.

    Whichever wins the registry lock, the durable outcome must be the same:
    the successful role's route, enrollment and audit line all survive, and the
    rolled-back role leaves nothing behind — including its exact prior config
    bytes and permission mode.
    """
    import multiprocessing as mp_lib

    home = tmp_path / "concurrent"
    hermes = home / ".hermes"
    profiles_root = hermes / "profiles"
    failing, surviving = "raphael-planner", "raphael-business"
    for role in (failing, surviving):
        (profiles_root / role).mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("HERMES_HOME", str(hermes))

    # The failing role starts with a config a managed deployment would leave:
    # different bytes AND a different mode from what a guarded write produces.
    prior = profiles_root / failing / "config.yaml"
    prior.write_text(
        yaml.safe_dump({"display": {"skin": "slate"}}), encoding="utf-8"
    )
    prior.chmod(0o640)
    prior_text = prior.read_text(encoding="utf-8")

    barrier_dir = tmp_path / "barrier"
    barrier_dir.mkdir()
    ctx = mp_lib.get_context("spawn")
    outs = {}
    procs = []
    for index, (role, fail) in enumerate(((failing, True), (surviving, False))):
        outs[role] = tmp_path / f"out-{role}.json"
        procs.append(ctx.Process(
            target=_concurrent_batch_worker,
            args=(role, str(barrier_dir), index, 2, fail, str(outs[role])),
        ))
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join(timeout=120)

    # No deadlock, and no worker retried anything.
    assert [proc.is_alive() for proc in procs] == [False, False]
    assert [proc.exitcode for proc in procs] == [0, 0]
    assert json.loads(outs[surviving].read_text(encoding="utf-8"))["ok"] is True
    failed = json.loads(outs[failing].read_text(encoding="utf-8"))
    assert failed["ok"] is False and failed["status"] == 409

    # Registry: only the role that actually committed.
    assert model_policy.enrolled_profile_ids() == frozenset({surviving})

    # Configs: the survivor's exact admitted, fallback-free route is live.
    # Read back through the canonical validator, which is what actually decides
    # whether a role is on an admitted route.
    assert configured_assignment_for(surviving) == assignment_for(
        surviving, "anthropic"
    )
    # ...and the rolled-back role is byte-for-byte, mode-for-mode as it was.
    assert prior.read_text(encoding="utf-8") == prior_text
    assert stat.S_IMODE(prior.stat().st_mode) == 0o640

    # The audit trail says exactly what happened: the survivor, nobody else.
    assert _audited_roles() == [surviving]
