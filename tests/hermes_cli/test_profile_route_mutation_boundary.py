"""Every profile-config write goes through one model-route mutation boundary.

``hermes_cli.config.profile_route_write`` is the single locked seam every
surface reaches — the Models endpoint, ``/api/model/set``, ``/api/config``,
``/api/config/raw``, ``hermes config set``, ``/model``, ``/reasoning``, the TUI
and setup — so the read, the compare-and-swap, policy validation, the
legacy-work fence and the write are one serialized decision rather than one per
caller.

Whether a profile is governed is EXPLICIT, persisted enrollment — never an
inference from the route the profile currently runs.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os

import pytest
import yaml

from hermes_cli import config as hermes_config
from hermes_cli import kanban_db, owner_workspace as ow
from hermes_cli.profiles import get_profile_dir, profile_name_for_home
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from plugins.dashboard_auth.raphael_workspace import model_policy

_ADMITTED = {
    "model": {"provider": "anthropic", "default": "claude-opus-5"},
    "agent": {"reasoning_effort": "max"},
    "fallback_providers": [],
}
_VERIFIER_ADMITTED = {
    "model": {"provider": "openai-codex", "default": "gpt-5.6-sol"},
    "agent": {"reasoning_effort": "max"},
    "fallback_providers": [],
}


def _scoped_save(profile: str, config: dict, **kwargs):
    directory = get_profile_dir(profile)
    directory.mkdir(parents=True, exist_ok=True)
    token = set_hermes_home_override(str(directory))
    try:
        hermes_config.save_config(dict(config), **kwargs)
    finally:
        reset_hermes_home_override(token)
    return directory


def _written(profile: str) -> dict:
    text = (get_profile_dir(profile) / "config.yaml").read_text(encoding="utf-8")
    return yaml.safe_load(text) or {}


def _revision(profile: str) -> str:
    from hermes_cli.web_routers.profiles import _profile_route_revision

    return _profile_route_revision(get_profile_dir(profile))


def test_profile_name_is_recovered_from_the_resolved_config_path():
    assert profile_name_for_home(get_profile_dir("raphael-builder")) == (
        "raphael-builder"
    )
    assert profile_name_for_home(get_profile_dir("default")) == "default"


# ---------------------------------------------------------------------------
# Explicit enrollment (never inferred from the current route)
# ---------------------------------------------------------------------------


def test_enrollment_is_explicit_and_persisted():
    assert model_policy.enrolled_profile_ids() == frozenset()
    model_policy.enroll_profile("raphael-verifier")
    assert model_policy.is_profile_enrolled("raphael-verifier")
    assert not model_policy.is_profile_enrolled("default")
    # Idempotent, and durable across readers.
    model_policy.enroll_profile("raphael-verifier")
    assert model_policy.enrolled_profile_ids() == frozenset({"raphael-verifier"})


def test_only_admitted_profile_ids_can_be_enrolled():
    with pytest.raises(ValueError, match="unadmitted Raphael profile id"):
        model_policy.enroll_profile("elias")
    assert model_policy.enrolled_profile_ids() == frozenset()


def test_an_unreadable_enrollment_record_fails_closed():
    path = model_policy.enrollment_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(model_policy.EnrollmentUnavailable):
        model_policy.enrolled_profile_ids()
    # And the write boundary refuses rather than assuming nothing is governed.
    with pytest.raises(RuntimeError, match="enrollment record"):
        _scoped_save("raphael-verifier", _VERIFIER_ADMITTED)


def test_an_enrollment_record_naming_an_unadmitted_profile_is_ignored():
    path = model_policy.enrollment_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({
            "version": model_policy.ENROLLMENT_SCHEMA_VERSION,
            "profiles": {"elias": {"authority": model_policy.ENROLLMENT_AUTHORITY}},
        }),
        encoding="utf-8",
    )
    assert model_policy.enrolled_profile_ids() == frozenset()


def test_an_enrolled_profile_stays_governed_even_when_it_has_drifted():
    """Governance survives drift — that is the whole point of persisting it."""
    # Drift the profile OFF any admitted route while unenrolled (native).
    _scoped_save("raphael-verifier", {"model": {"provider": "openai", "default": "gpt-x"}})
    model_policy.enroll_profile("raphael-verifier")
    # Route inference would now conclude "not on the policy, so not governed".
    with pytest.raises(ValueError, match="unadmitted Raphael model assignment"):
        _scoped_save(
            "raphael-verifier", {"model": {"provider": "openai", "default": "gpt-y"}}
        )
    # An admitted route is still accepted, which is how a drifted role recovers.
    _scoped_save("raphael-verifier", _VERIFIER_ADMITTED)
    assert _written("raphael-verifier")["model"]["default"] == "gpt-5.6-sol"


def test_an_unenrolled_default_profile_on_the_same_model_stays_native():
    """Same name AND same model as a policy role, but never enrolled."""
    _scoped_save("default", _ADMITTED)
    assert _written("default")["model"]["default"] == "claude-opus-5"
    # Unrestricted: it can move anywhere, including off the policy's route.
    _scoped_save("default", {"model": {"provider": "openai", "default": "gpt-x"}})
    assert _written("default")["model"]["default"] == "gpt-x"
    # Forbidden markers are a policy concept; an ordinary profile is not judged.
    _scoped_save("default", {"model": {"provider": "anthropic", "default": "claude-fable-5"}})
    assert _written("default")["model"]["default"] == "claude-fable-5"


def test_non_policy_profiles_keep_unrestricted_native_behaviour():
    _scoped_save("elias", {"model": {"provider": "openai", "default": "gpt-x"}})
    assert _written("elias")["model"]["default"] == "gpt-x"


# ---------------------------------------------------------------------------
# Policy validation at the boundary
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "override",
    [
        # A model the policy does not admit for this role.
        {"model": {"provider": "openai-codex", "default": "gpt-5.6-terra"}},
        # A forbidden model, whatever the role.
        {"model": {"provider": "anthropic", "default": "claude-fable-5"}},
        # A forbidden effort.
        {"agent": {"reasoning_effort": "ultra"}},
        # Fallbacks re-enabled: the route is no longer pinned at all.
        {"fallback_providers": [{"provider": "openai", "model": "gpt-x"}]},
        {"fallback_model": "gpt-x"},
    ],
)
def test_unadmitted_route_is_refused_at_the_shared_boundary(override):
    """Any surface reaching the boundary is fenced, not just the Models token."""
    _scoped_save("raphael-verifier", _VERIFIER_ADMITTED)
    model_policy.enroll_profile("raphael-verifier")
    with pytest.raises(ValueError, match="unadmitted Raphael model assignment"):
        _scoped_save("raphael-verifier", {**_VERIFIER_ADMITTED, **override})
    # The admitted route survived; nothing half-applied.
    assert _written("raphael-verifier")["model"]["default"] == "gpt-5.6-sol"


def test_a_deep_task_lane_is_not_a_valid_base_route():
    """Base configs use validate_assignment; deep lanes are task authority only.

    ``raphael-builder``/anthropic admits ``claude-sonnet-5`` as its BASE route
    and ``claude-opus-5`` only as its deep TASK lane. Accepting the deep lane
    as a base route would erase the routine/deep separation.
    """
    base = {
        "model": {"provider": "anthropic", "default": "claude-sonnet-5"},
        "agent": {"reasoning_effort": "max"},
        "fallback_providers": [],
    }
    _scoped_save("raphael-builder", base)
    model_policy.enroll_profile("raphael-builder")
    deep = model_policy.task_assignment_for("raphael-builder", "anthropic", "deep")
    assert deep.model == "claude-opus-5"
    with pytest.raises(ValueError, match="unadmitted Raphael model assignment"):
        _scoped_save("raphael-builder", {**base, "model": {
            "provider": "anthropic", "default": deep.model,
        }})
    assert _written("raphael-builder")["model"]["default"] == "claude-sonnet-5"
    # The same route IS admitted as a task pin.
    model_policy.validate_runtime_assignment(
        "raphael-builder", "anthropic", deep.model, "max", disable_fallbacks=True,
    )


def test_unrelated_config_writes_do_not_enter_the_guard(monkeypatch):
    """A write that does not move the route must not fence or serialize."""
    _scoped_save("raphael-verifier", _VERIFIER_ADMITTED)
    model_policy.enroll_profile("raphael-verifier")

    def _never(*args, **kwargs):  # pragma: no cover - asserted by not running
        raise AssertionError("an unrelated config write must not fence routes")

    monkeypatch.setattr(ow, "fence_effective_task_routes", _never)
    _scoped_save("raphael-verifier", {**_VERIFIER_ADMITTED, "agent": {
        "reasoning_effort": "max", "max_turns": 40,
    }})
    assert _written("raphael-verifier")["agent"]["max_turns"] == 40


def test_route_change_fences_existing_owner_work_before_writing(monkeypatch):
    """The fence runs inside the guard, ahead of the write, and fails closed."""
    _scoped_save("default", _ADMITTED)
    model_policy.enroll_profile("default")
    calls: list[str] = []

    def _boom(profile):
        calls.append(profile)
        raise ow.OwnerWorkspaceError(
            "execution_state_busy", "existing owner work could not be read",
        )

    monkeypatch.setattr(ow, "fence_effective_task_routes", _boom)
    with pytest.raises(ow.OwnerWorkspaceError):
        _scoped_save("default", {
            "model": {"provider": "openai-codex", "default": "gpt-5.6-sol"},
            "agent": {"reasoning_effort": "max"},
            "fallback_providers": [],
        })

    assert calls == ["default"]
    # The old route survived: an unfenceable change is not half-applied.
    assert _written("default")["model"]["provider"] == "anthropic"


# ---------------------------------------------------------------------------
# The one shared single-key writer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("round_trip", [False, True])
def test_single_key_writers_reach_the_same_boundary(round_trip):
    """`config set`, /model, /reasoning and the TUI cannot bypass the policy."""
    directory = _scoped_save("raphael-verifier", _VERIFIER_ADMITTED)
    model_policy.enroll_profile("raphael-verifier")
    path = directory / "config.yaml"
    with pytest.raises(ValueError, match="unadmitted Raphael model assignment"):
        hermes_config.save_shared_config_key(
            path, "model.default", "gpt-5.6-terra", round_trip=round_trip,
        )
    assert _written("raphael-verifier")["model"]["default"] == "gpt-5.6-sol"
    # An unrelated key still writes normally.
    hermes_config.save_shared_config_key(
        path, "display.timestamps", True, round_trip=round_trip,
    )
    assert _written("raphael-verifier")["display"]["timestamps"] is True


def test_single_key_writer_leaves_ordinary_profiles_alone():
    directory = _scoped_save("elias", {"model": {"provider": "openai", "default": "a"}})
    hermes_config.save_shared_config_key(
        directory / "config.yaml", "model.default", "b",
    )
    assert _written("elias")["model"]["default"] == "b"


def test_config_set_reaches_the_boundary(monkeypatch):
    """`hermes config set` writes through the guard, not a bare YAML dump."""
    directory = _scoped_save("raphael-verifier", _VERIFIER_ADMITTED)
    model_policy.enroll_profile("raphael-verifier")
    token = set_hermes_home_override(str(directory))
    try:
        with pytest.raises(ValueError, match="unadmitted Raphael model assignment"):
            hermes_config.set_config_value("model.default", "gpt-5.6-terra")
    finally:
        reset_hermes_home_override(token)
    assert _written("raphael-verifier")["model"]["default"] == "gpt-5.6-sol"


# ---------------------------------------------------------------------------
# Compare-and-swap
# ---------------------------------------------------------------------------


def test_route_revision_changes_only_with_the_effective_route():
    _scoped_save("raphael-verifier", _VERIFIER_ADMITTED)
    first = _revision("raphael-verifier")
    _scoped_save("raphael-verifier", {**_VERIFIER_ADMITTED, "agent": {
        "reasoning_effort": "max", "max_turns": 40,
    }})
    assert _revision("raphael-verifier") == first

    _scoped_save("raphael-verifier", {
        "model": {"provider": "anthropic", "default": "claude-opus-5"},
        "agent": {"reasoning_effort": "max"},
        "fallback_providers": [],
    })
    assert _revision("raphael-verifier") != first


def test_save_config_returns_the_revision_it_wrote():
    directory = get_profile_dir("elias")
    directory.mkdir(parents=True, exist_ok=True)
    token = set_hermes_home_override(str(directory))
    try:
        revision = hermes_config.save_config({
            "model": {"provider": "openai", "default": "gpt-x"},
        })
    finally:
        reset_hermes_home_override(token)
    assert revision == _revision("elias")


def test_conditional_write_rejects_a_stale_revision():
    _scoped_save("elias", {"model": {"provider": "openai", "default": "gpt-x"}})
    current = _revision("elias")
    # Matching revision: accepted.
    _scoped_save(
        "elias",
        {"model": {"provider": "openai", "default": "gpt-y"}},
        expected_revision=current,
    )
    assert _written("elias")["model"]["default"] == "gpt-y"
    # The revision the caller held is now stale.
    with pytest.raises(hermes_config.RouteRevisionConflict, match="revision changed"):
        _scoped_save(
            "elias",
            {"model": {"provider": "openai", "default": "gpt-z"}},
            expected_revision=current,
        )
    assert _written("elias")["model"]["default"] == "gpt-y"


# --- real cross-process compare-and-swap -----------------------------------


def _cas_child(home: str, profile: str, revision: str, target: str, queue):
    """Run one CAS write in a genuinely separate OS process."""
    os.environ["HERMES_HOME"] = home
    try:
        from hermes_cli import config as child_config
        from hermes_cli.profiles import get_profile_dir as child_profile_dir
        from hermes_constants import set_hermes_home_override as child_override

        child_override(str(child_profile_dir(profile)))
        child_config.save_config(
            {"model": {"provider": "openai", "default": target}},
            expected_revision=revision,
        )
        queue.put(("ok", target))
    except child_config.RouteRevisionConflict:
        queue.put(("conflict", target))
    except BaseException as exc:  # pragma: no cover - diagnostic only
        queue.put(("error", f"{type(exc).__name__}: {exc}"))


def test_two_processes_using_one_revision_yield_one_success_one_conflict():
    """Real cross-process CAS: the lock, the read and the compare are one step.

    Both children state the SAME revision. Whichever acquires this profile's
    kernel-held route lock first writes; the other then reads a revision that
    no longer matches and is refused. Exactly one of each, never two writes.
    """
    profile = "elias"
    _scoped_save(profile, {"model": {"provider": "openai", "default": "gpt-start"}})
    revision = _revision(profile)
    home = os.environ["HERMES_HOME"]

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    children = [
        ctx.Process(
            target=_cas_child, args=(home, profile, revision, f"gpt-{n}", queue)
        )
        for n in ("a", "b")
    ]
    for child in children:
        child.start()
    for child in children:
        child.join(timeout=120)
        assert child.exitcode == 0

    outcomes = [queue.get(timeout=30) for _ in children]
    statuses = sorted(status for status, _ in outcomes)
    assert statuses == ["conflict", "ok"], outcomes
    winner = next(target for status, target in outcomes if status == "ok")
    assert _written(profile)["model"]["default"] == winner


# ---------------------------------------------------------------------------
# End-to-end fence
# ---------------------------------------------------------------------------


def test_fence_pins_owner_work_before_a_real_route_change(monkeypatch):
    """End to end: a route change through save_config freezes existing work."""
    profile = "default"
    _scoped_save(profile, _ADMITTED)
    model_policy.enroll_profile(profile)
    with kanban_db.connect() as conn:
        # Classified but not yet locked — the state a pre-lock owner commit
        # leaves behind, and the only one the policy can mint authority for.
        # (An unclassifiable route is paused instead; that path is covered by
        # ``test_an_unpinnable_receipt_owned_task_is_paused_not_left_runnable``.)
        task_id = kanban_db.create_task(
            conn,
            title="owner work",
            assignee=profile,
            execution_tier="routine",
            project_id="p_owner",
        )
    # Stand in for the receipt store: this id is proven owner-created work on
    # the default board.
    monkeypatch.setattr(
        ow, "_owner_receipt_task_ids", lambda: {kanban_db.DEFAULT_BOARD: {task_id}},
    )

    _scoped_save(profile, {
        "model": {"provider": "openai-codex", "default": "gpt-5.6-sol"},
        "agent": {"reasoning_effort": "max"},
        "fallback_providers": [],
    })

    with kanban_db.connect() as conn:
        pinned = kanban_db.get_task(conn, task_id)
    # Frozen on the route it was already running, not on the new selection.
    assert (pinned.provider_override, pinned.model_override) == (
        "anthropic", "claude-opus-5",
    )
    assert kanban_db.policy_lock_error(
        pinned.model_policy_lock, profile, "anthropic", "claude-opus-5", "max",
        "routine",
    ) is None
    assert _written(profile)["model"]["provider"] == "openai-codex"
