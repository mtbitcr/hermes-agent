"""Owner-approved model version updates (model update flow plan, card 1).

The admitted route matrix is hard-coded in ``model_policy``. One small record
under the ROOT Hermes home, ``raphael/model_policy_updates.json``, lets an
owner Decision move a model to its next version without a code edit (D1). One
entry is one Decision and covers every role that uses that model (D2).

The record is a list of owner decisions, never a model catalog: an entry moves
only the version of a model inside the same provider. It never names a role, an
effort or a second provider, an entry without a decision reference is ignored,
and a record that cannot be read leaves the hard-coded matrix in force and
refuses every write.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli.dashboard_auth.audit import read_audit_records
from hermes_constants import get_default_hermes_root
from plugins.dashboard_auth.raphael_workspace import model_policy
from plugins.dashboard_auth.raphael_workspace.model_policy import (
    EnrollmentUnavailable,
    admitted_profile_ids,
    admitted_provider_ids,
    assignment_for,
    mint_policy_lock,
    policy_lock_error,
    project_options_payload,
    task_assignment_for,
)

_TIERS = ("routine", "deep")

# One owner Decision each, as ``(provider, from_model, to_model)``.
_OPUS = ("anthropic", "claude-opus-5-5", "claude-opus-6")
_SONNET = ("anthropic", "claude-sonnet-5-5", "claude-sonnet-6")
_SOL = ("openai-codex", "gpt-6.1-sol", "gpt-6.2-sol")


def _record_path() -> Path:
    """Where the record lives: under the ROOT home, never a profile home."""
    return get_default_hermes_root() / "raphael" / "model_policy_updates.json"


def _entry(
    update,
    *,
    state="active",
    decision_ref="decision-1",
    approved_at="2026-11-16T09:00:00Z",
    **fields,
) -> dict:
    provider, from_model, to_model = update
    return {
        "provider": provider,
        "from_model": from_model,
        "to_model": to_model,
        "decision_ref": decision_ref,
        "approved_at": approved_at,
        "state": state,
        **fields,
    }


# A well-formed Decision recorded beside every entry the policy must ignore, so
# "ignored" is proven per entry and the record is known to have been read.
_CONTROL = _entry(_SOL, decision_ref="decision-control")


def _write_record(*entries) -> Path:
    path = _record_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"version": 1, "updates": list(entries)}), encoding="utf-8"
    )
    return path


def _routes() -> dict:
    """Every admitted ``(role, provider, tier)`` route the policy resolves now."""
    routes = {}
    for profile in admitted_profile_ids():
        for provider in admitted_provider_ids():
            try:
                assignment_for(profile, provider)
            except ValueError:
                continue
            for tier in _TIERS:
                routes[(profile, provider, tier)] = task_assignment_for(
                    profile, provider, tier
                )
    return routes


def _with_control_applied(routes: dict) -> dict:
    """``routes`` after the control Decision alone."""
    provider, from_model, to_model = _SOL
    return {
        key: (
            replace(route, model=to_model, model_label=to_model)
            if key[1] == provider and route.model == from_model
            else route
        )
        for key, route in routes.items()
    }


def _stored_lock(assignee, provider, model, effort, tier) -> str:
    """A lock in the stored format, from the documented canonical form."""
    canonical = json.dumps(
        {
            "authority": "raphael",
            "version": 1,
            "assignee": assignee,
            "provider": provider,
            "model": model,
            "reasoning_effort": effort,
            "execution_tier": tier,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return "raphael:v1:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _snapshot(path: Path):
    if path.is_dir():
        return ("directory", sorted(child.name for child in path.iterdir()))
    return path.read_bytes()


# ---------------------------------------------------------------------------
# An active entry replaces the version everywhere the model is used
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("update", [_OPUS, _SONNET, _SOL], ids=lambda u: u[2])
def test_active_entry_moves_every_role_and_tier_that_used_from_model(update):
    provider, from_model, to_model = update
    before = _routes()
    users = {
        key
        for key, route in before.items()
        if key[1] == provider and route.model == from_model
    }
    assert users, "the hard-coded matrix must use from_model somewhere"

    _write_record(_entry(update))

    after = _routes()
    assert after.keys() == before.keys()
    for key, route in before.items():
        if key in users:
            # Only the version moves: same role, provider, effort and
            # recommendation, and the route is named by its exact model id
            # rather than keeping the old model's label.
            assert after[key] == replace(route, model=to_model, model_label=to_model)
        else:
            assert after[key] == route
    for profile, provider_id, tier in after:
        if tier == "routine":
            assert assignment_for(profile, provider_id) == after[
                (profile, provider_id, tier)
            ]


def test_options_projection_offers_to_model_in_place_of_from_model():
    native = {
        "providers": [{
            "slug": "anthropic",
            "authenticated": True,
            "models": ["claude-opus-5-5", "claude-opus-6", "claude-sonnet-5-5"],
        }],
    }
    _write_record(_entry(_OPUS))

    def _offered(profile):
        (row,) = project_options_payload(native, profile=profile)["providers"]
        return row["models"]

    assert _offered("raphael-planner") == ["claude-opus-6"]
    # The builder's routine lane is Sonnet; only its Opus deep lane moved.
    assert _offered("raphael-builder") == ["claude-opus-6", "claude-sonnet-5-5"]
    assert _offered(None) == ["claude-opus-6", "claude-sonnet-5-5"]


# ---------------------------------------------------------------------------
# Locks: the old route keeps verifying and can never be minted again
# ---------------------------------------------------------------------------

_SEALED_LANES = (
    ("raphael-planner", "routine"),
    ("raphael-planner", "deep"),
    ("raphael-builder", "deep"),
)


def test_lock_minted_on_from_model_before_the_entry_still_validates_and_cannot_be_minted():
    provider, from_model, to_model = _OPUS
    # Seals minted while from_model was current, at the base effort and at a
    # risk-tier pinned effort.
    sealed = [
        (mint_policy_lock(*route), route)
        for route in (
            (profile, provider, from_model, effort, tier)
            for profile, tier in _SEALED_LANES
            for effort in ("max", "high")
        )
    ]

    _write_record(_entry(_OPUS))

    for lock, route in sealed:
        assert policy_lock_error(lock, *route) is None
        with pytest.raises(ValueError):
            mint_policy_lock(*route)
        assignee, _provider, _model, effort, tier = route
        moved = (assignee, provider, to_model, effort, tier)
        assert policy_lock_error(mint_policy_lock(*moved), *moved) is None
    # The retired route verifies only where it was used: never on a role,
    # provider or tier that did not run from_model.
    for stray in (
        ("raphael-business", provider, from_model, "high", "routine"),
        ("raphael-planner", "openai-codex", from_model, "max", "routine"),
    ):
        assert policy_lock_error(_stored_lock(*stray), *stray) is not None


def test_rolled_back_entry_restores_from_model_and_to_model_locks_still_validate():
    provider, from_model, to_model = _OPUS
    before = _routes()
    active = _entry(_OPUS)
    model_policy.apply_model_update_record(active)
    sealed = [
        (mint_policy_lock(*route), route)
        for route in (
            (profile, provider, to_model, effort, tier)
            for profile, tier in _SEALED_LANES
            for effort in ("max", "high")
        )
    ]

    # The Undo marks the same entry rolled back.
    model_policy.apply_model_update_record({**active, "state": "rolled_back"})

    assert _routes() == before
    assert list(model_policy.read_model_updates()) == [
        {**active, "state": "rolled_back"}
    ]
    for lock, route in sealed:
        assert policy_lock_error(lock, *route) is None
        with pytest.raises(ValueError):
            mint_policy_lock(*route)
    for profile, tier in _SEALED_LANES:
        restored = (profile, provider, from_model, "max", tier)
        assert policy_lock_error(mint_policy_lock(*restored), *restored) is None


def test_a_later_decision_moves_the_version_an_earlier_one_introduced():
    provider = "anthropic"
    before = _routes()
    first = _entry(_OPUS)
    second = _entry(
        (provider, "claude-opus-6", "claude-opus-7"), decision_ref="decision-2"
    )
    sealed = []
    for decision, model in ((first, "claude-opus-6"), (second, "claude-opus-7")):
        model_policy.apply_model_update_record(decision)
        route = ("raphael-planner", provider, model, "high", "deep")
        sealed.append((mint_policy_lock(*route), route))

    moved = _routes()
    for key, route in before.items():
        if key[1] == provider and route.model == "claude-opus-5-5":
            assert moved[key].model == "claude-opus-7"
        else:
            assert moved[key] == route

    # Undone latest first, the matrix is back and every sealed lock verifies.
    model_policy.apply_model_update_record({**second, "state": "rolled_back"})
    assert assignment_for("raphael-planner", provider).model == "claude-opus-6"
    model_policy.apply_model_update_record({**first, "state": "rolled_back"})
    assert _routes() == before
    for lock, route in sealed:
        assert policy_lock_error(lock, *route) is None
        with pytest.raises(ValueError):
            mint_policy_lock(*route)


# ---------------------------------------------------------------------------
# The record's boundary: only a version, inside the same provider
# ---------------------------------------------------------------------------

_OUT_OF_BOUNDS = {
    "forbidden-marker": _entry(("anthropic", "claude-opus-5-5", "claude-opus-fable-6")),
    "forbidden-marker-any-case": _entry(
        ("anthropic", "claude-opus-5-5", "Claude-Opus-6-UltraCode")
    ),
    "unadmitted-provider": _entry(("openrouter", "claude-opus-5-5", "claude-opus-6")),
    "provider-that-never-ran-the-model": _entry(
        ("openai-codex", "claude-opus-5-5", "claude-opus-6")
    ),
    "other-provider-model": _entry(("anthropic", "claude-opus-5-5", "gpt-7-sol")),
    "provider-change": _entry(_OPUS, to_provider="openai-codex"),
    "effort-change": _entry(_OPUS, reasoning_effort="high"),
    "role-named": _entry(_OPUS, profile="raphael-planner"),
    "other-model-not-a-version": _entry(
        ("anthropic", "claude-sonnet-5-5", "claude-opus-6")
    ),
    "variant-not-a-version": _entry(
        ("anthropic", "claude-opus-5-5", "claude-opus-5-5-fast")
    ),
    "unknown-state": _entry(_OPUS, state="pending"),
}


@pytest.mark.parametrize("case", sorted(_OUT_OF_BOUNDS))
def test_out_of_bounds_entry_is_ignored_and_hard_coded_route_stays(case):
    before = _routes()
    entry = _OUT_OF_BOUNDS[case]
    # Recorded active AND rolled back: honoured either way it would move a
    # route or make its to_model verify.
    recorded = [entry]
    if entry["state"] == "active":
        recorded.append(
            {**entry, "decision_ref": "decision-2", "state": "rolled_back"}
        )
    path = _write_record(*recorded, _CONTROL)

    assert _routes() == _with_control_applied(before)
    for (profile, provider, tier), route in before.items():
        stray = (profile, provider, entry["to_model"], route.reasoning_effort, tier)
        assert policy_lock_error(_stored_lock(*stray), *stray) is not None
        with pytest.raises(ValueError):
            mint_policy_lock(*stray)

    # The writer refuses what the reader ignores, and writes nothing.
    prior = path.read_bytes()
    with pytest.raises(ValueError):
        model_policy.apply_model_update_record(entry)
    assert path.read_bytes() == prior


_NO_DECISION = {
    "missing": {
        key: value for key, value in _entry(_OPUS).items() if key != "decision_ref"
    },
    "null": _entry(_OPUS, decision_ref=None),
    "empty": _entry(_OPUS, decision_ref=""),
    "blank": _entry(_OPUS, decision_ref="   "),
    "not-text": _entry(_OPUS, decision_ref=7),
}


@pytest.mark.parametrize("case", sorted(_NO_DECISION))
def test_entry_without_decision_ref_is_ignored(case):
    before = _routes()
    entry = _NO_DECISION[case]
    path = _write_record(entry, _CONTROL)

    assert _routes() == _with_control_applied(before)
    with pytest.raises(ValueError):
        mint_policy_lock("raphael-planner", "anthropic", "claude-opus-6", "max", "routine")

    prior = path.read_bytes()
    with pytest.raises(ValueError):
        model_policy.apply_model_update_record(entry)
    assert path.read_bytes() == prior


# ---------------------------------------------------------------------------
# An unreadable record fails closed
# ---------------------------------------------------------------------------

_UNREADABLE = {
    "not-json": lambda path: path.write_bytes(b"\xff\xfe{ not json"),
    "directory": lambda path: path.mkdir(),
    "unknown-version": lambda path: path.write_text(
        json.dumps({"version": 2, "updates": [_entry(_OPUS)]}), encoding="utf-8"
    ),
    "updates-not-a-list": lambda path: path.write_text(
        json.dumps({"version": 1, "updates": {"0": _entry(_OPUS)}}), encoding="utf-8"
    ),
    "not-an-object": lambda path: path.write_text(
        json.dumps([_entry(_OPUS)]), encoding="utf-8"
    ),
}


@pytest.mark.parametrize("case", sorted(_UNREADABLE))
def test_unreadable_record_fails_closed_and_refuses_writes(case):
    before = _routes()
    path = _write_record(_entry(_OPUS))
    on_update = ("raphael-planner", "anthropic", "claude-opus-6", "max", "routine")
    assert assignment_for("raphael-planner", "anthropic").model == "claude-opus-6"
    lock = mint_policy_lock(*on_update)

    path.unlink()
    _UNREADABLE[case](path)
    unreadable = _snapshot(path)

    # The hard-coded matrix applies, and nothing the record admitted still
    # verifies or mints.
    assert _routes() == before
    assert policy_lock_error(lock, *on_update) is not None
    with pytest.raises(ValueError):
        mint_policy_lock(*on_update)

    # Every write refuses and leaves the record exactly as it is.
    with pytest.raises(EnrollmentUnavailable):
        model_policy.read_model_updates()
    for update in (
        _entry(_SONNET, decision_ref="decision-2"),
        _entry(_OPUS, state="rolled_back"),
    ):
        with pytest.raises(EnrollmentUnavailable):
            model_policy.apply_model_update_record(update)
    assert _snapshot(path) == unreadable


# ---------------------------------------------------------------------------
# The writer: one Decision, one entry, under the enrollment lock
# ---------------------------------------------------------------------------


def test_writes_hold_the_enrollment_lock_and_restore_puts_back_the_exact_prior_text(
    monkeypatch,
):
    held = []
    real_lock = model_policy._enrollment_registry_lock

    @contextlib.contextmanager
    def _spy():
        with real_lock() as locked:
            held.append(True)
            yield locked

    monkeypatch.setattr(model_policy, "_enrollment_registry_lock", _spy)

    def _locked(write, *args):
        count = len(held)
        result = write(*args)
        assert len(held) > count, "a record write ran outside the enrollment lock"
        return result

    path = _record_path()
    # No record yet: the prior text is None, and restoring it removes the file.
    snapshot = _locked(model_policy.apply_model_update_record, _entry(_OPUS))
    assert snapshot is None and path.is_file()
    _locked(model_policy.restore_model_update_record, snapshot)
    assert not path.exists()

    # An existing record comes back byte for byte.
    prior = json.dumps(
        {"version": 1, "updates": [_entry(_SOL, decision_ref="decision-sol")]},
        indent=4,
    ) + "\n"
    path.write_text(prior, encoding="utf-8")
    snapshot = _locked(model_policy.apply_model_update_record, _entry(_OPUS))
    assert snapshot == prior
    assert [entry["decision_ref"] for entry in model_policy.read_model_updates()] == [
        "decision-sol",
        "decision-1",
    ]
    _locked(model_policy.restore_model_update_record, snapshot)
    assert path.read_bytes() == prior.encode("utf-8")


def test_one_decision_is_one_entry_that_only_moves_from_active_to_rolled_back():
    active = _entry(_OPUS)
    model_policy.apply_model_update_record(active)
    # Re-applying the same Decision changes nothing.
    model_policy.apply_model_update_record(active)
    for refused in (
        # The same Decision cannot be re-pointed at another model.
        _entry(_SONNET),
        # A Decision never recorded as active cannot be rolled back.
        _entry(_SONNET, decision_ref="decision-2", state="rolled_back"),
    ):
        with pytest.raises(ValueError):
            model_policy.apply_model_update_record(refused)

    model_policy.apply_model_update_record({**active, "state": "rolled_back"})
    # Applying it again takes a new Decision.
    with pytest.raises(ValueError):
        model_policy.apply_model_update_record(active)
    assert list(model_policy.read_model_updates()) == [
        {**active, "state": "rolled_back"}
    ]


# ---------------------------------------------------------------------------
# The Models audit names the model
# ---------------------------------------------------------------------------


def _models_request(path: str):
    return SimpleNamespace(
        method="PUT",
        url=SimpleNamespace(path=path),
        client=SimpleNamespace(host="127.0.0.1"),
        headers={},
        state=SimpleNamespace(
            token_authenticated=True,
            token_route_template=path,
            token_principal=SimpleNamespace(
                provider="raphael-models-token",
                principal="raphael-models",
                credential_id="cred-1",
            ),
        ),
    )


def test_models_audit_entry_carries_model():
    model_policy.audit_models_machine_success(
        _models_request("/api/profiles/raphael-planner/model"),
        action="assign",
        profile="raphael-planner",
        provider="anthropic",
    )
    _write_record(_entry(_OPUS))
    model_policy.journal_models_machine_batch_success(
        _models_request("/api/profiles/model-batch"),
        action="assign",
        roles=[
            ("raphael-planner", "anthropic"),
            ("raphael-builder", "anthropic"),
            ("raphael-verifier", "openai-codex"),
        ],
    ).commit()
    model_policy.audit_models_machine_success(
        _models_request("/api/providers/oauth"),
        action="providers",
        profile="raphael-planner",
    )

    audited = [
        (record["profile"], record["model_provider"], record.get("model", "absent"))
        for record in read_audit_records()
        if record.get("source") == "raphael-models"
    ]
    assert audited == [
        ("raphael-planner", "anthropic", "claude-opus-5-5"),
        ("raphael-planner", "anthropic", "claude-opus-6"),
        ("raphael-builder", "anthropic", "claude-sonnet-5-5"),
        ("raphael-verifier", "openai-codex", "gpt-6.1-sol"),
        ("raphael-planner", None, None),
    ]


# ---------------------------------------------------------------------------
# A dispatched worker reads the ROOT record
# ---------------------------------------------------------------------------


def test_worker_environment_reads_the_record_under_the_root_home(monkeypatch):
    from hermes_cli.kanban_db import kanban_db_path

    root = get_default_hermes_root()
    board = "raphael-models"
    worker = "raphael-builder"
    _write_record(_entry(_SONNET))
    profile_home = root / "profiles" / worker
    # A record under the profile home is not the policy's and is never read.
    decoy = profile_home / "raphael" / "model_policy_updates.json"
    decoy.parent.mkdir(parents=True)
    decoy.write_text(
        json.dumps({
            "version": 1,
            "updates": [_entry(
                ("anthropic", "claude-sonnet-5-5", "claude-sonnet-7"),
                decision_ref="decision-decoy",
            )],
        }),
        encoding="utf-8",
    )

    # Exactly what the dispatcher injects into a worker it spawns.
    monkeypatch.setenv("HERMES_KANBAN_DB", str(kanban_db_path(board=board)))
    monkeypatch.setenv("HERMES_KANBAN_BOARD", board)
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_worker_1")
    monkeypatch.setenv("HERMES_HOME", str(profile_home))

    assert get_default_hermes_root() == root
    assert assignment_for(worker, "anthropic").model == "claude-sonnet-6"
    assert task_assignment_for(worker, "anthropic", "routine").model == "claude-sonnet-6"
    route = (worker, "anthropic", "claude-sonnet-6", "max", "routine")
    assert policy_lock_error(mint_policy_lock(*route), *route) is None
    assert [entry["decision_ref"] for entry in model_policy.read_model_updates()] == [
        "decision-1"
    ]
