"""No writer of ``assignee`` can turn required-review work into its own reviewer.

``create_task`` refuses to attach a committed review requirement to a
read-only reviewer card, because such a card IS the independent audit and
would otherwise become its own reviewer. That refusal is worth nothing if the
same state can be reached one call later: assign implementation work that
carries ``requires_review`` to the read-only reviewer profile, and the row ends
up ``assignee=<reviewer>``, ``owned_paths=[]``, ``requires_review=true`` —
exactly the shape creation calls impossible, whose handover then parks with
implementer == reviewer.

The refusal is narrow on purpose, and both edges are asserted here:

* it applies while the card is IMPLEMENTATION work (carries the requirement
  and is not on the genuine review lane); and
* the committed requirement is never downgraded to make the assignment legal —
  refusing is the answer, silently clearing an owner's committed specification
  is not.

The two things that must keep working are asserted alongside it: explicit
assignment on the genuine ``review`` lane, and the pre-existing read-only audit
review card that carries no requirement at all.

**And the rule is a LIFETIME invariant, not a property of one entry point.**
It used to live only in ``assign_task``, so three other writers of the same
column reached straight past it — the triage specification, the triage
decomposition's root flip, and the dispatcher's ``kanban.default_assignee``
fallback. Each of them also asked the route authority and then DISCARDED the
route it handed back, writing the assignee alone, which left governed work
pinned to the role it no longer had for its next claim to be refused by. Every
one of those writers is exercised below against both an ordinary row (which
must still be assigned, with the route and scope the assignment is entitled to)
and a review-required / owner-governed one (which must be refused, with nothing
written and the row's route still naming the role that holds it).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from plugins.dashboard_auth.raphael_workspace import model_policy


READ_ONLY_REVIEWER = "raphael-verifier"
# A reviewer the policy can nominate that is NOT read-only, so the
# owner-governed cases below are decided by the route lock rather than by the
# read-only exclusion — the two failures are separate and are asserted apart.
MUTATING_REVIEWER = "raphael-planner"
GOVERNED_OWNER = "default"


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
    """Make the model policy nominate exactly these reviewer roles."""
    monkeypatch.setattr(
        model_policy, "reviewer_profile_ids", lambda: tuple(profiles), raising=True,
    )


def _resolvable_profiles(monkeypatch):
    """Let the dispatcher treat every profile name as a real Hermes profile."""
    monkeypatch.setattr(
        "hermes_cli.profiles.profile_exists", lambda name: True, raising=True,
    )


def _spawn(*args, **kwargs):
    """Stand-in for the real worker spawn — returns a fake PID."""
    return 12345


def _govern(conn, task_id: str, *, assignee: str = GOVERNED_OWNER) -> None:
    """Make the row receipt-owned and pin it under a valid route lock."""
    route = model_policy.task_assignment_for(assignee, "anthropic", "routine")
    lock = kb.mint_policy_lock(
        assignee, route.provider, route.model, route.reasoning_effort, "routine",
    )
    conn.execute(
        "UPDATE tasks SET owner_receipt_bound = 1, execution_tier = 'routine', "
        "provider_override = ?, model_override = ?, reasoning_effort = ?, "
        "model_policy_lock = ? WHERE id = ?",
        (route.provider, route.model, route.reasoning_effort, lock, task_id),
    )
    conn.commit()
    row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
    assert kb.task_is_policy_governed(row) is True
    assert kb.task_policy_lock_error(row) is None


_CHILDREN = [{"title": "do the work", "parents": []}]


# ---------------------------------------------------------------------------
# The same invariant, from the three writers that used to reach past it
# ---------------------------------------------------------------------------


def test_specify_assigns_an_ordinary_triage_row_with_its_scope(kanban_home):
    """CONTROL: specification still assigns, and the scope rides with the role.

    Ordinary triage work with no committed requirement can still be specified
    onto the read-only reviewer — that is how the audit review card is made —
    and the boundary that role is entitled to is written in the SAME statement
    as the assignee, not left for a later claim to discover.
    """
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="rough idea", triage=True)
        assert kb.specify_triage_task(
            conn, tid, title="Audit the thing", assignee=READ_ONLY_REVIEWER,
        ) is True
        specified = kb.get_task(conn, tid)
        assert specified.assignee == READ_ONLY_REVIEWER
        assert specified.status in ("todo", "ready")
        assert specified.owned_paths == []
        assert specified.integrates_parent_heads is False


def test_specify_refuses_review_required_work_to_the_read_only_reviewer(
    kanban_home,
):
    """THE BYPASS: the triage specification path reached past the invariant.

    Triage implementation work that carries the committed review requirement
    could be specified straight onto the read-only reviewer. Its scope is
    erased at claim time, and its own handover then parks an unassigned review
    with no head — the exact row ``create_task`` refuses, reached through a
    door that never asked.
    """
    with kb.connect() as conn:
        tid = kb.create_task(
            conn, title="rough idea", assignee="worker",
            requires_review=True, triage=True,
        )
        with pytest.raises(RuntimeError, match="requires_review"):
            kb.specify_triage_task(
                conn, tid, title="Implement the thing",
                assignee=READ_ONLY_REVIEWER,
            )
        refused = kb.get_task(conn, tid)
        assert refused.status == "triage", "a refused specification writes nothing"
        assert refused.assignee == "worker"
        assert refused.requires_review is True
        assert _events(conn, tid, "specified") == []

        # Not a blanket refusal: the same specification lands on an
        # implementation profile, and the requirement survives it.
        assert kb.specify_triage_task(
            conn, tid, title="Implement the thing", assignee="other-worker",
        ) is True
        specified = kb.get_task(conn, tid)
        assert specified.assignee == "other-worker"
        assert specified.requires_review is True


def test_decompose_assigns_an_ordinary_root_with_its_scope(kanban_home):
    """CONTROL: the decomposed root is still assigned, scope and all."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="ship a feature", triage=True)
        child_ids = kb.decompose_triage_task(
            conn, tid, root_assignee=READ_ONLY_REVIEWER, children=_CHILDREN,
        )
        assert child_ids is not None and len(child_ids) == 1
        root = kb.get_task(conn, tid)
        assert root.assignee == READ_ONLY_REVIEWER
        assert root.status in ("todo", "ready")
        assert root.owned_paths == []


def test_decompose_refuses_a_review_required_root_to_the_read_only_reviewer(
    kanban_home,
):
    """THE BYPASS, from the decomposition's root flip."""
    with kb.connect() as conn:
        tid = kb.create_task(
            conn, title="ship a reviewed feature", assignee="worker",
            requires_review=True, triage=True,
        )
        with pytest.raises(RuntimeError, match="requires_review"):
            kb.decompose_triage_task(
                conn, tid, root_assignee=READ_ONLY_REVIEWER, children=_CHILDREN,
            )
        refused = kb.get_task(conn, tid)
        assert refused.status == "triage"
        assert refused.assignee == "worker"
        assert refused.requires_review is True
        # The whole fan-out aborts with the root: no orphan children.
        assert _events(conn, tid, "decomposed") == []
        assert kb.child_ids(conn, tid) == []

        # Not a blanket refusal: an implementation root decomposes fine.
        assert kb.decompose_triage_task(
            conn, tid, root_assignee="orchestrator", children=_CHILDREN,
        ) is not None
        assert kb.get_task(conn, tid).assignee == "orchestrator"


def test_the_default_assignee_adopts_an_ordinary_row_with_its_scope(
    kanban_home, monkeypatch,
):
    """CONTROL: the dispatcher's fallback still adopts unassigned work."""
    _resolvable_profiles(monkeypatch)
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="unassigned work", assignee=None)
        result = kb.dispatch_once(
            conn, spawn_fn=_spawn, dry_run=False,
            default_assignee=READ_ONLY_REVIEWER,
        )
        assert result.auto_assigned_default == [tid]
        adopted = kb.get_task(conn, tid)
        assert adopted.assignee == READ_ONLY_REVIEWER
        assert adopted.owned_paths == []
        assert _events(conn, tid, "assigned")[0]["source"] == (
            "kanban.default_assignee"
        )


def test_the_default_assignee_refuses_review_required_work_and_governed_work(
    kanban_home, monkeypatch,
):
    """THE BYPASS, from the dispatcher's own fallback assignment.

    Two rows the fallback may not adopt, for two different reasons, and
    neither of them may be adopted QUIETLY: implementation work carrying the
    committed review requirement (the read-only reviewer would become its own
    reviewer), and owner-governed work (the fallback is not an owner approval,
    so it cannot hand receipt-owned work to a role the owner never approved).
    """
    _resolvable_profiles(monkeypatch)
    with kb.connect() as conn:
        reviewed = kb.create_task(
            conn, title="unassigned reviewed work", assignee=None,
            requires_review=True,
        )
        governed = kb.create_task(conn, title="unassigned owner work", assignee=None)
        conn.execute(
            "UPDATE tasks SET owner_receipt_bound = 1 WHERE id = ?", (governed,),
        )
        conn.commit()

        result = kb.dispatch_once(
            conn, spawn_fn=_spawn, dry_run=False,
            default_assignee=READ_ONLY_REVIEWER,
        )
        assert result.auto_assigned_default == []
        assert set(result.skipped_unassigned) == {reviewed, governed}
        for tid in (reviewed, governed):
            row = kb.get_task(conn, tid)
            assert row.assignee is None, "an unadoptable row stays unassigned"
            assert row.status != "running"
            assert _events(conn, tid, "assigned") == []
        assert kb.get_task(conn, reviewed).requires_review is True


def test_no_assignee_writer_can_strand_governed_work_off_its_route_lock(
    kanban_home, monkeypatch,
):
    """The second consequence: the route must never be left behind.

    Each of these writers asked the route authority and then DISCARDED what it
    handed back, writing only the assignee. On owner-governed work whose
    committed specification carries the review requirement, that authority used
    to authorize itself — the target merely had to be the profile the policy
    nominates — so the row changed hands while provider, model, effort, tier
    and lock went on describing the role that no longer held it, and the next
    claim was refused by the task's own lock.

    The authority is the review round-trip's own now, and neither of these is a
    leg of it. Whatever the outcome, the invariant asserted here is the one
    that matters: a governed row's route still names the role that holds it, so
    it can still start a run.
    """
    _nominate(monkeypatch, MUTATING_REVIEWER)
    assert kb.policy_resolved_reviewer() == MUTATING_REVIEWER, (
        "this test proves nothing unless the policy really nominates it"
    )
    with kb.connect() as conn:
        specified = kb.create_task(
            conn, title="governed, reviewed, specified", assignee=GOVERNED_OWNER,
            requires_review=True, triage=True,
        )
        decomposed = kb.create_task(
            conn, title="governed, reviewed, decomposed", assignee=GOVERNED_OWNER,
            requires_review=True, triage=True,
        )
        for tid in (specified, decomposed):
            _govern(conn, tid)

        with pytest.raises(RuntimeError):
            kb.specify_triage_task(
                conn, specified, title="spec", assignee=MUTATING_REVIEWER,
            )
        with pytest.raises(RuntimeError):
            kb.decompose_triage_task(
                conn, decomposed, root_assignee=MUTATING_REVIEWER,
                children=_CHILDREN,
            )

        for tid in (specified, decomposed):
            row = kb.get_task(conn, tid)
            assert row.assignee == GOVERNED_OWNER, (
                "a refused role change may not move the assignee"
            )
            assert row.status == "triage"
            # The load-bearing invariant: route and role still agree, so the
            # row can still start a run under the owner's approved authority.
            kb.assert_claimable_route(conn, tid)

        # And the refusal is about the ROLE CHANGE, not about these writers:
        # specifying without moving the role promotes the governed row and it
        # claims cleanly on its owner-approved route.
        assert kb.specify_triage_task(
            conn, specified, title="spec, same role",
        ) is True
        assert kb.claim_task(
            conn, specified, claimer=f"{GOVERNED_OWNER}:1",
        ) is not None


def test_assigning_required_review_work_to_the_reviewer_is_refused(kanban_home):
    """The PUBLIC assign-then-handover bypass, end to end."""
    with kb.connect() as conn:
        tid = kb.create_task(
            conn, title="implementation work", assignee="worker",
            requires_review=True,
        )
        before = kb.get_task(conn, tid)
        assert before.requires_review is True

        with pytest.raises(RuntimeError, match="requires_review"):
            kb.assign_task(conn, tid, READ_ONLY_REVIEWER)

        after = kb.get_task(conn, tid)
        assert after.assignee == "worker"
        assert after.requires_review is True, (
            "the committed requirement must never be cleared to make the "
            "assignment legal"
        )
        assert after.owned_paths == before.owned_paths
        assert after.integrates_parent_heads == before.integrates_parent_heads
        assert _events(conn, tid, "assigned") == []

        # And the handover that follows still parks under an INDEPENDENT
        # reviewer rather than under the implementer itself.
        assert _hand_over(conn, tid) is True
        parked = kb.get_task(conn, tid)
        assert parked.status == "review"
        requested = _events(conn, tid, "review_requested")
        assert len(requested) == 1
        assert requested[0]["implementer"] == "worker"
        assert requested[0]["reviewer"] != requested[0]["implementer"], (
            "a parked card whose reviewer is its implementer is self-review"
        )
        assert parked.assignee != "worker"


def test_the_genuine_review_lane_still_accepts_an_explicit_reviewer(kanban_home):
    """REGRESSION: the refusal must not reach the real review handoff."""
    with kb.connect() as conn:
        tid = kb.create_task(
            conn, title="parked work", assignee="worker", requires_review=True,
        )
        assert _hand_over(conn, tid) is True
        assert kb.get_task(conn, tid).status == "review"

        # A human re-routing the review to the read-only reviewer while the
        # card sits on the genuine review lane is the supported action.
        assert kb.assign_task(conn, tid, READ_ONLY_REVIEWER) is True
        routed = kb.get_task(conn, tid)
        assert routed.assignee == READ_ONLY_REVIEWER
        assert routed.status == "review"
        assert routed.requires_review is True


def test_the_read_only_audit_review_card_can_still_be_assigned(kanban_home):
    """REGRESSION: work with no committed requirement is untouched.

    Reassigning ordinary work to the read-only reviewer profile still
    succeeds and still converts the scope to read-only.
    """
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="audit this", assignee="worker")
        assert kb.get_task(conn, tid).requires_review is False

        assert kb.assign_task(conn, tid, READ_ONLY_REVIEWER) is True
        audit = kb.get_task(conn, tid)
        assert audit.assignee == READ_ONLY_REVIEWER
        assert audit.owned_paths == []
        assert audit.integrates_parent_heads is False
