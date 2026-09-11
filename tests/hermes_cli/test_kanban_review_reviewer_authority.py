"""WHO reviews is a policy decision, and an unresolved reviewer fails closed.

A committed review requirement says only THAT the work is independently
reviewed. The reviewer *identity* is therefore not the kernel's to invent: it
is asked of the team's model policy at the moment of the handover, through the
policy-owned selector :func:`reviewer_profile_ids`. Two properties are load
bearing and are asserted here against the REAL handover and the REAL
dispatcher:

* the selected identity FOLLOWS the policy — when the policy nominates a
  different reviewer role, a different profile ends up holding the parked card
  (not merely a different provider or model for the same role); and
* it FAILS CLOSED — when no independent reviewer resolves (empty nomination,
  a roster that admits none of them, or a nomination that is the implementer
  itself), the kernel parks the work with NO assignee. It never selects the
  implementer, because the review dispatcher would then re-claim the card for
  the very profile that wrote the code and call that an independent review.

A policy module that exposes no selector, or one whose selector cannot be
read, fails closed the same way: the work is parked with no assignee. The
kernel never falls back to its own read-only reviewer registry.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from plugins.dashboard_auth.raphael_workspace import model_policy


READ_ONLY_REVIEWER = "raphael-verifier"
SELECTOR = "reviewer_profile_ids"


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return home


def _events(conn, tid, kind) -> list:
    return [
        json.loads(r["payload"]) if r["payload"] else None
        for r in conn.execute(
            "SELECT payload FROM task_events WHERE task_id = ? AND kind = ? "
            "ORDER BY id",
            (tid, kind),
        ).fetchall()
    ]


def _hand_over(conn, task_id, *, claimer="worker:1", summary="ready for review"):
    """The REAL implementer handover: claim the task, then complete it."""
    run = kb.claim_task(conn, task_id, claimer=claimer)
    assert run is not None
    return kb.complete_task(
        conn, task_id, summary=summary, expected_run_id=run.current_run_id,
    )


def _nominate(monkeypatch, *profiles):
    """Make the model policy nominate exactly these reviewer roles.

    Uses ``raising=True`` because the selector is a real, shipped attribute
    on the policy module — :func:`reviewer_profile_ids` — so monkeypatching
    must confirm the target exists rather than inventing it.
    """
    monkeypatch.setattr(
        model_policy, SELECTOR, lambda: tuple(profiles), raising=True,
    )


def _dispatch_capturing(conn):
    """One REAL dispatcher tick; returns (result, [(task_id, assignee), ...])."""
    spawned: list[tuple[str, object]] = []

    def spawn(task, workspace):
        spawned.append((task.id, task.assignee))
        return None

    result = kb.dispatch_once(conn, spawn_fn=spawn)
    return result, spawned


def test_the_reviewer_identity_follows_the_policy_owned_selector(
    kanban_home, monkeypatch, all_assignees_spawnable
):
    """A different policy answer means a different reviewer PROFILE.

    Nothing about the two cards differs except what the policy nominated at
    the moment each handover happened.
    """
    with kb.connect() as conn:
        _nominate(monkeypatch, READ_ONLY_REVIEWER)
        first = kb.create_task(
            conn, title="first", assignee="worker", requires_review=True,
        )
        assert _hand_over(conn, first) is True
        assert kb.get_task(conn, first).assignee == READ_ONLY_REVIEWER

        # The policy now names a different reviewer role.
        _nominate(monkeypatch, "raphael-planner")
        second = kb.create_task(
            conn, title="second", assignee="worker", requires_review=True,
        )
        assert _hand_over(conn, second, claimer="worker:2") is True

        parked = kb.get_task(conn, second)
        assert parked.status == "review"
        assert parked.assignee == "raphael-planner", (
            "the reviewer IDENTITY must be the one the policy selector "
            "nominates, not a constant the kernel holds"
        )
        requested = _events(conn, second, "review_requested")
        assert len(requested) == 1
        assert requested[0]["implementer"] == "worker"
        assert requested[0]["reviewer"] == "raphael-planner"

        # The real dispatcher hands the parked card to that reviewer.
        _, spawned = _dispatch_capturing(conn)
        assert (second, "raphael-planner") in spawned


def test_the_shipped_policy_module_exposes_the_reviewer_selector(kanban_home):
    """The policy module owns the reviewer identity — the selector MUST exist.

    ``model_policy.reviewer_profile_ids()`` is the single source of truth for
    which role(s) perform independent review. The kernel calls it on every
    handover and intersects its answer with the admitted roster. Today that is
    the read-only audit reviewer; tomorrow the policy could name a different
    roster without any kernel change.
    """
    assert hasattr(model_policy, SELECTOR), (
        "the shipped policy module must expose the reviewer selector — "
        "the kernel's fail-closed path is for ABSENT or BROKEN selectors, "
        "not an alternative to shipping one"
    )
    assert callable(getattr(model_policy, SELECTOR))
    result = model_policy.reviewer_profile_ids()
    assert isinstance(result, tuple)
    assert READ_ONLY_REVIEWER in result

    with kb.connect() as conn:
        tid = kb.create_task(
            conn, title="shipped policy", assignee="worker", requires_review=True,
        )
        assert _hand_over(conn, tid) is True
        assert kb.get_task(conn, tid).assignee == READ_ONLY_REVIEWER


def test_an_unresolvable_reviewer_is_never_the_implementer(
    kanban_home, monkeypatch, all_assignees_spawnable
):
    """FAIL CLOSED: no reviewer resolves -> the card parks with no assignee.

    The roster admits no read-only reviewer role and the policy nominates
    nobody, so there is no independent reviewer. The requirement is still
    honoured — the card parks on the review lane instead of completing — but
    the implementer must not be handed its own work back by the dispatcher.
    """
    with kb.connect() as conn:
        tid = kb.create_task(
            conn, title="nobody reviews today", assignee="worker",
            requires_review=True,
        )
        monkeypatch.setattr(
            model_policy,
            "admitted_profile_ids",
            lambda: tuple(
                p for p in model_policy._PROFILE_IDS if p not in kb._READ_ONLY_PROFILES
            ),
        )
        assert kb.policy_resolved_reviewer() is None

        assert _hand_over(conn, tid) is True
        parked = kb.get_task(conn, tid)
        assert parked.status == "review", "the requirement is still honoured"
        assert parked.completed_at is None
        assert parked.assignee is None, (
            "with no independent reviewer the kernel must park the work "
            "unassigned, never on the implementer"
        )
        assert _events(conn, tid, "review_requested")[0]["implementer"] == "worker"

        # The REAL dispatcher selection path: nothing is spawned for the
        # implementer, and the card stays parked for a human to route.
        result, spawned = _dispatch_capturing(conn)
        assert spawned == [], (
            f"the parked card must not be dispatched to anyone, got {spawned}"
        )
        assert tid in result.skipped_unassigned
        still = kb.get_task(conn, tid)
        assert still.status == "review"
        assert still.assignee is None


def test_a_reviewer_that_equals_the_implementer_is_no_reviewer_at_all(
    kanban_home, monkeypatch, all_assignees_spawnable
):
    """Self-review is not review, even when the policy nominates it."""
    with kb.connect() as conn:
        _nominate(monkeypatch, "raphael-builder")
        tid = kb.create_task(
            conn, title="its own reviewer", assignee="raphael-builder",
            requires_review=True,
        )
        assert kb.policy_resolved_reviewer() == "raphael-builder"

        assert _hand_over(conn, tid, claimer="raphael-builder:1") is True
        parked = kb.get_task(conn, tid)
        assert parked.status == "review"
        assert parked.assignee is None, (
            "a reviewer that IS the implementer is not an independent "
            "reviewer, so the handover must fail closed"
        )
        _, spawned = _dispatch_capturing(conn)
        assert spawned == []


def test_a_nomination_outside_the_admitted_roster_resolves_nobody(
    kanban_home, monkeypatch
):
    """The nomination is intersected with the admitted roster, not trusted."""
    with kb.connect() as conn:
        _nominate(monkeypatch, "not-a-profile")
        assert kb.policy_resolved_reviewer() is None

        tid = kb.create_task(
            conn, title="unadmitted nominee", assignee="worker",
            requires_review=True,
        )
        assert _hand_over(conn, tid) is True
        parked = kb.get_task(conn, tid)
        assert parked.status == "review"
        assert parked.assignee is None


def test_kernel_read_only_profiles_constant_has_no_reviewer_identity_effect(
    kanban_home, monkeypatch
):
    """The kernel's _READ_ONLY_PROFILES is NOT a reviewer-identity fallback.

    The constant names roles that may never own repository writes (used by the
    write-scope guards in assign_task and create_task), but it is NOT consulted
    for reviewer identity. That decision belongs entirely to the policy's
    reviewer_profile_ids() selector.

    Reproduction: if mutating _READ_ONLY_PROFILES changed the resolved reviewer,
    the kernel — not the policy — would be deciding who reviews. After the fix,
    the selector alone controls the identity.
    """
    original_reviewer = kb.policy_resolved_reviewer()
    assert original_reviewer == READ_ONLY_REVIEWER, (
        "baseline: the shipped policy nominates the read-only reviewer"
    )

    monkeypatch.setattr(kb, "_READ_ONLY_PROFILES", frozenset({"raphael-planner"}))

    after_mutation = kb.policy_resolved_reviewer()
    assert after_mutation == READ_ONLY_REVIEWER, (
        f"mutating _READ_ONLY_PROFILES must NOT change the resolved reviewer; "
        f"got {after_mutation!r} instead of {READ_ONLY_REVIEWER!r}. "
        f"The selector — not the kernel constant — decides who reviews."
    )


def test_a_selector_that_raises_resolves_nobody(kanban_home, monkeypatch):
    """An unreadable nomination is not a licence to pick a kernel reviewer.

    If the policy's selector raises, the kernel fails closed — it returns None
    rather than inventing a reviewer of its own choosing.
    """
    def broken_selector():
        raise RuntimeError("transient policy failure")

    monkeypatch.setattr(model_policy, SELECTOR, broken_selector, raising=True)
    assert kb.policy_resolved_reviewer() is None, (
        "a selector that raises must resolve None (fail closed), "
        "never a kernel-chosen fallback"
    )


def test_an_absent_selector_resolves_nobody_not_a_kernel_identity(
    kanban_home, monkeypatch, all_assignees_spawnable
):
    """A policy with no selector fails closed — the kernel invents nobody.

    The reviewer identity is the POLICY's decision. If the selector is absent
    (deleted, not-yet-shipped, or deliberately removed for a lockdown), the
    kernel returns None, and the handover parks the card with no assignee. It
    never falls back to _READ_ONLY_PROFILES or any other kernel-owned constant.
    """
    monkeypatch.delattr(model_policy, SELECTOR, raising=True)
    assert not hasattr(model_policy, SELECTOR), "sanity: selector is gone"

    assert kb.policy_resolved_reviewer() is None, (
        "an absent selector must resolve None (fail closed), "
        "never a kernel-chosen fallback like _READ_ONLY_PROFILES"
    )

    with kb.connect() as conn:
        tid = kb.create_task(
            conn, title="no selector", assignee="worker", requires_review=True,
        )
        assert _hand_over(conn, tid) is True
        parked = kb.get_task(conn, tid)
        assert parked.status == "review", "the requirement is still honoured"
        assert parked.assignee is None, (
            "with no selector the card must park unassigned, not under a "
            "kernel-invented reviewer identity"
        )

        _, spawned = _dispatch_capturing(conn)
        assert spawned == [], (
            "unassigned review cards must not dispatch to anyone"
        )
