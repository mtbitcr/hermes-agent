"""Every new card runs at the effort and time box its risk tier and kind call for.

P2 of the risk tier plan (tests P2-1 to P2-6), through the real kernel:

* effort is pinned at creation and sealed in the card's existing route lock:
  high by default, max for tier 2 and for a security review card (R12); a
  new governed card without a tier counts as tier 2 and records it, so a
  review round trip keeps its pin, while a card from before keeps its own
  effort and time box (decision 3);
* the time box is pinned at creation from the kind of work (decision 2): an
  integration card gets the integration box, and a review run gets the review
  box while its card keeps its own;
* the pins use only what the model policy admits: high next to max on the
  lane's own model, nothing wider, and never a fallback;
* every new governed build card is reviewed (decision 4), so a build run past
  its box with one saved patch parks in review instead of being retried;
* ``kanban_create`` carries ``risk_tier`` on an owner-governed board, under
  the exact environment the dispatcher gives a worker.
"""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import closing
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_constants import VALID_REASONING_EFFORTS
from plugins.dashboard_auth.raphael_workspace import model_policy as mp

# The owner-workspace harness (authorized proposal context, auto-approver,
# canonical payloads) is reused verbatim, so the owner-created cards here go
# through the REAL kernel entry points.
from tests.hermes_cli.test_owner_workspace import (  # noqa: F401
    _bootstrap_board,
    _commit_project_plan,
    _commit_task_graph,
    _configured_provider,
    _project_plan_args,
    _task_graph_args,
    _with_approver,
    ctx,
)
from tests.hermes_cli._kanban_fence_support import (
    DEFAULT_GIT_IDENTITY, create_fenced_board, git, make_git_repo,
)
from tests.hermes_cli.test_kanban_committed_review_requirement import _finding

_PROVIDER = {"raphael-verifier": "openai-codex"}
BUILD_BOX = 7200
DEEP_BOX = 2700
ROUTINE_BOX = 1800


@pytest.fixture
def kanban_home(fence_home):
    kb.init_db()
    return fence_home


def _row(conn, task_id: str) -> sqlite3.Row:
    return conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()


def _locked_card(
    conn, assignee: str, execution_tier: str, *, risk_tier, provider=None,
    responsibility=None, owned_paths=None, **extra,
) -> str:
    """Create a card the way the owner kernel does: the policy's route, sealed."""
    provider = provider or _PROVIDER.get(assignee, "anthropic")
    route = mp.task_assignment_for(assignee, provider, execution_tier)
    extra.setdefault("workspace_kind", "worktree" if owned_paths else "scratch")
    return kb.create_task(
        conn,
        title=f"{assignee} {execution_tier} tier {risk_tier}",
        assignee=assignee,
        responsibility=responsibility,
        execution_tier=execution_tier,
        provider_override=route.provider,
        model_override=route.model,
        reasoning_effort=route.reasoning_effort,
        model_policy_lock=kb.mint_policy_lock(
            assignee, route.provider, route.model, route.reasoning_effort,
            execution_tier,
        ),
        owned_paths=owned_paths,
        risk_tier=risk_tier,
        **extra,
    )


def _add(title: str, tier: int, responsibility: str = "B03") -> dict:
    return {
        "action": "add",
        "reason": "Create one bounded owner-approved task.",
        "title": title,
        "body": "Produce the owner-visible result.",
        "assignee": "default",
        "responsibility": responsibility,
        "execution_tier": "routine",
        "risk_tier": tier,
        "existing_parents": [],
        "new_parents": [],
    }


def _owner_created_rows(ctx, path: str, tiers, responsibilities) -> list:
    """Commit one approved proposal creating one task per tier; return its rows."""
    titles = [f"Task {index}" for index in range(len(tiers))]
    if path == "task_graph":
        args = _task_graph_args(
            idempotency_key=f"graph-pins-{'-'.join(map(str, tiers))}-"
            f"{'-'.join(responsibilities)}",
            project_name="Pinned Effort Project",
        )
        template = args["tasks"][0]
        args["tasks"] = [
            {
                **template, "title": title, "responsibility": responsibility,
                "risk_tier": tier, "parents": [],
            }
            for title, tier, responsibility in zip(titles, tiers, responsibilities)
        ]
        approver = _with_approver(ctx.session)
        try:
            result = _commit_task_graph(ctx, **args)
        finally:
            approver.join()
        board, created = result["board"], result["task_ids"]
    else:
        setup = _bootstrap_board(ctx)
        changes = [
            _add(title, tier, responsibility)
            for title, tier, responsibility in zip(titles, tiers, responsibilities)
        ]
        approver = _with_approver(ctx.session)
        try:
            result = _commit_project_plan(
                ctx,
                **_project_plan_args(
                    setup, changes,
                    idempotency_key=f"plan-pins-{'-'.join(map(str, tiers))}-"
                    f"{'-'.join(responsibilities)}",
                ),
            )
        finally:
            approver.join()
        board, created = setup["board"], result["created_task_ids"]
    assert result["ok"] is True
    with kb.connect(board=board) as conn:
        return [_row(conn, task_id) for task_id in created]


def _kanban_create(args: dict) -> dict:
    """Create a card through the registered, schema-checked ``kanban_create``."""
    from jsonschema import Draft202012Validator

    from tools import kanban_tools  # noqa: F401  (registers the tool)
    from tools.registry import registry

    schema = registry.get_entry("kanban_create").schema["parameters"]
    Draft202012Validator(schema).validate(args)
    return json.loads(registry.dispatch("kanban_create", args))


# ---------------------------------------------------------------------------
# P2-1 and P2-2: the pinned effort
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", ["task_graph", "project_plan"])
def test_effort_is_pinned_high_by_default_and_max_for_tier_2(ctx, path):
    rows = _owner_created_rows(ctx, path, (0, 1, 2), ("B03", "B03", "B03"))

    assert [(row["risk_tier"], row["reasoning_effort"]) for row in rows] == [
        (0, "high"), (1, "high"), (2, "max"),
    ]
    for row in rows:
        # The pin is sealed: the same lock the dispatcher re-validates.
        assert row["model_policy_lock"]
        assert kb.task_policy_lock_error(row) is None
        assert row["model_override"] == "claude-opus-5-5"


@pytest.mark.parametrize("path", ["task_graph", "project_plan"])
@pytest.mark.parametrize("tier", [0, 1])
def test_a_security_review_card_is_pinned_at_max(ctx, path, tier):
    rows = _owner_created_rows(ctx, path, (tier, tier), ("B03", "R12"))

    # Decision 6: only "Security review" (R12) raises the effort below tier 2.
    assert [
        (row["responsibility"], row["risk_tier"], row["reasoning_effort"])
        for row in rows
    ] == [("B03", tier, "high"), ("R12", tier, "max")]
    for row in rows:
        assert kb.task_policy_lock_error(row) is None


# ---------------------------------------------------------------------------
# P2-3: the pinned time box
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("assignee", "execution_tier", "owned_paths", "box"), [
    # Build work, a mutating scope: two hours, routine or deep.
    ("raphael-claude-worker", "deep", ["src"], BUILD_BOX),
    ("raphael-claude-worker", "routine", ["src"], BUILD_BOX),
    # Read-only work: analysis when deep, a proposal when routine.
    ("raphael-claude-worker", "deep", [], DEEP_BOX),
    ("raphael-planner", "deep", [], DEEP_BOX),
    ("raphael-business", "routine", [], ROUTINE_BOX),
    ("raphael-designer", "routine", [], ROUTINE_BOX),
    # Review cards: 45 minutes (decision 2).
    ("raphael-verifier", "routine", None, DEEP_BOX),
    ("raphael-verifier", "deep", None, DEEP_BOX),
    # Release, integration and infrastructure cards: 45 minutes per run.
    ("raphael-builder", "deep", ["."], DEEP_BOX),
    ("raphael-builder", "routine", ["."], DEEP_BOX),
    # Coordinator cards: 45 minutes when deep, 30 when routine.
    ("default", "deep", None, DEEP_BOX),
    ("default", "routine", None, ROUTINE_BOX),
])
def test_the_time_box_is_pinned_at_creation(
    kanban_home, assignee, execution_tier, owned_paths, box,
):
    with kb.connect() as conn:
        task_id = _locked_card(
            conn, assignee, execution_tier, risk_tier=1, owned_paths=owned_paths,
        )
        row = _row(conn, task_id)
        task = kb.get_task(conn, task_id)

    assert row["max_runtime_seconds"] == box
    assert task.max_runtime_seconds == box
    assert kb.task_policy_lock_error(row) is None


def test_an_integration_card_gets_the_integration_box(kanban_home):
    """The existing marker makes integration work, whichever profile runs it."""
    with kb.connect() as conn:
        rows = [
            _row(conn, _locked_card(
                conn, "raphael-claude-worker", "deep", risk_tier=2,
                owned_paths=["."], integrates_parent_heads=integrates,
            ))
            for integrates in (True, False)
        ]

    assert [
        (row["integrates_parent_heads"], row["max_runtime_seconds"]) for row in rows
    ] == [(1, DEEP_BOX), (0, BUILD_BOX)]
    for row in rows:
        assert kb.task_policy_lock_error(row) is None


def test_a_card_without_a_route_lock_is_not_pinned(kanban_home):
    """CLI and legacy cards carry no approved route, so nothing is pinned."""
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn, title="CLI build", assignee="raphael-claude-worker",
            workspace_kind="worktree", owned_paths=["src"], risk_tier=1,
        )
        row = _row(conn, task_id)

    assert (row["reasoning_effort"], row["max_runtime_seconds"]) == (None, None)


# ---------------------------------------------------------------------------
# P2-4: the pin is sealed
# ---------------------------------------------------------------------------


def test_a_pinned_effort_cannot_be_edited_silently(kanban_home):
    with kb.connect() as conn:
        routine = _locked_card(
            conn, "raphael-claude-worker", "deep", risk_tier=1, owned_paths=["src"],
        )
        top = _locked_card(
            conn, "raphael-claude-worker", "deep", risk_tier=2, owned_paths=["src"],
        )
        for task_id, pinned, edited in ((routine, "high", "max"), (top, "max", "high")):
            row = _row(conn, task_id)
            assert row["reasoning_effort"] == pinned
            assert kb.task_policy_lock_error(row) is None

            conn.execute(
                "UPDATE tasks SET reasoning_effort = ? WHERE id = ?",
                (edited, task_id),
            )

            assert kb.task_policy_lock_error(_row(conn, task_id)) is not None


@pytest.mark.parametrize(("tier", "responsibility", "effort"), [
    (1, None, "high"),  # decision 6: code review stays high below tier 2
    (1, "R12", "max"),  # a security review card keeps max
    (2, None, "max"),  # decision 1: a raise to tier 2 re-pins max
])
def test_the_review_handover_re_pins_the_tier_effort(
    kanban_home, tier, responsibility, effort,
):
    with kb.connect() as conn:
        task_id = _locked_card(
            conn, "raphael-claude-worker", "deep", risk_tier=1, owned_paths=["src"],
            responsibility=responsibility, requires_review=True,
        )
        conn.execute("UPDATE tasks SET risk_tier = ? WHERE id = ?", (tier, task_id))
        reviewer = kb.policy_resolved_reviewer()
        assignments, repin = kb.role_transition_route(
            conn, task_id, reviewer, review_round_trip=True,
        )

    route = dict(assignments, assignee=reviewer)
    assert reviewer == "raphael-verifier"
    assert route["reasoning_effort"] == repin["reasoning_effort"] == effort
    assert kb.task_policy_lock_error(route) is None


# ---------------------------------------------------------------------------
# P2-5: no other effort, model, route or fallback
# ---------------------------------------------------------------------------


def test_no_other_effort_model_route_or_fallback(kanban_home):
    """A relationship between data, over every admitted lane and tier."""
    pinned_efforts = {"high", "max"}
    with kb.connect() as conn:
        for profile, provider in sorted(mp._ASSIGNMENTS):
            base = mp.assignment_for(profile, provider)
            # The profile's own route is untouched: only its base effort, and
            # never with a fallback.
            with pytest.raises(ValueError):
                mp.validate_assignment(
                    profile, provider, base.model, base.reasoning_effort,
                    disable_fallbacks=False,
                )
            for other in pinned_efforts - {base.reasoning_effort}:
                with pytest.raises(ValueError):
                    mp.validate_assignment(
                        profile, provider, base.model, other, disable_fallbacks=True,
                    )
            for execution_tier in ("routine", "deep"):
                route = mp.task_assignment_for(profile, provider, execution_tier)
                for effort in set(VALID_REASONING_EFFORTS) - pinned_efforts:
                    with pytest.raises(ValueError):
                        kb.mint_policy_lock(
                            profile, provider, route.model, effort, execution_tier,
                        )
                for tier in (0, 1, 2):
                    for responsibility in (None, "R12"):
                        task_id = _locked_card(
                            conn, profile, execution_tier, risk_tier=tier,
                            provider=provider, responsibility=responsibility,
                            owned_paths=[],
                        )
                        row = _row(conn, task_id)
                        expected = (
                            "max" if tier == 2 or responsibility == "R12" else "high"
                        )
                        assert row["reasoning_effort"] in pinned_efforts
                        assert row["reasoning_effort"] == expected, (
                            profile, provider, execution_tier, tier, responsibility,
                        )
                        assert (row["provider_override"], row["model_override"]) == (
                            route.provider, route.model,
                        )
                        assert row["execution_tier"] == execution_tier
                        assert kb.task_policy_lock_error(row) is None


# ---------------------------------------------------------------------------
# P2-6: the build box hands over instead of retrying
# ---------------------------------------------------------------------------

SLUG = "pinned-box"
OWNED = "src/owned"
WORKER_PID = 70998
# What a build run saves: its patch, and possibly its report.
PATCH = (
    "x.patch",
    (
        f"diff --git a/{OWNED}/saved.py b/{OWNED}/saved.py\n"
        "new file mode 100644\n--- /dev/null\n"
        f"+++ b/{OWNED}/saved.py\n@@ -0,0 +1 @@\n+saved = True\n"
    ).encode("utf-8"),
    "text/x-diff",
)
REPORT = ("report.md", b"# Report\n\nSaved before the box ended.\n", "text/markdown")


@pytest.fixture
def build_repo(fence_home, tmp_path, monkeypatch):
    """A repository with an owned module, worked on from a governed board."""
    repo = tmp_path / "repo"
    make_git_repo(repo)
    (repo / OWNED).mkdir(parents=True)
    (repo / OWNED / "app.py").write_text("value = 1\n", encoding="utf-8")
    git(repo, "add", f"{OWNED}/app.py")
    git(repo, "commit", "-m", "owned module", author=DEFAULT_GIT_IDENTITY)
    create_fenced_board(SLUG, project_id="p_governed_test")
    monkeypatch.setenv("HERMES_KANBAN_BOARD", SLUG)
    monkeypatch.setenv("HERMES_KANBAN_ATTACHMENTS_ROOT", str(tmp_path / "at"))
    return repo


def _outlast(conn, task_id: str, run_id: int, box: int, pid: int) -> tuple:
    """Run past ``box`` as worker ``pid``; return the sweep's timeouts and signals."""
    kb._set_worker_pid(conn, task_id, pid)
    conn.execute(
        "UPDATE task_runs SET started_at = ? WHERE id = ?",
        (int(time.time()) - box - 60, run_id),
    )
    signalled = []
    timed_out = kb.enforce_max_runtime(
        conn, signal_fn=lambda target, _sig: signalled.append(target),
    )
    return timed_out, signalled


def _build_past_its_box(conn, task_id: str, saved) -> tuple:
    """Claim the build, save ``saved`` in its run, and run past the build box."""
    host = kb._claimer_id().split(":", 1)[0]
    claimed = kb.claim_task(conn, task_id, claimer=f"{host}:w0")
    workspace, branch = kb._resolve_worktree_workspace(claimed)
    kb.set_workspace_path(conn, task_id, workspace)
    kb.set_branch_name(conn, task_id, branch)
    kb.record_worktree_base(conn, task_id, workspace)
    run_id = claimed.current_run_id
    for name, data, content_type in saved:
        kb.store_attachment_bytes(
            conn, task_id, name, data, content_type=content_type,
            uploaded_by="agent", expected_run_id=run_id,
        )
    return (run_id, *_outlast(conn, task_id, run_id, BUILD_BOX, WORKER_PID))


def test_a_decomposed_build_child_is_reviewed(kanban_home):
    """Decision 4 reaches decomposed work too; a read-only child is not reviewed."""
    route = mp.task_assignment_for("raphael-claude-worker", "anthropic", "deep")
    child = {
        "assignee": "raphael-claude-worker", "execution_tier": "deep", "risk_tier": 1,
        "provider_override": route.provider, "model_override": route.model,
        "reasoning_effort": route.reasoning_effort,
        "model_policy_lock": kb.mint_policy_lock(
            "raphael-claude-worker", route.provider, route.model,
            route.reasoning_effort, "deep",
        ),
    }
    with kb.connect() as conn:
        root = kb.create_task(conn, title="Ship the owned module", triage=True)
        child_ids = kb.decompose_triage_task(
            conn, root, root_assignee="default", children=[
                {**child, "title": "Build", "owned_paths": [OWNED],
                 "workspace_kind": "worktree"},
                {**child, "title": "Analyse", "owned_paths": []},
            ],
        )
        rows = [_row(conn, task_id) for task_id in child_ids]

    assert [(row["requires_review"], row["max_runtime_seconds"]) for row in rows] == [
        (1, BUILD_BOX), (0, DEEP_BOX),
    ]
    for row in rows:
        assert kb.task_policy_lock_error(row) is None


@pytest.mark.parametrize(
    "saved", [(PATCH,), (PATCH, REPORT)], ids=["patch", "patch-and-report"],
)
def test_the_build_box_hands_over_instead_of_retrying(build_repo, saved):
    # P2-6 as planned: one saved patch is enough. The card is created through
    # the registered tool, and nobody asks for its review.
    created = _kanban_create({
        "title": "Build the owned module", "assignee": "raphael-claude-worker",
        "execution_tier": "deep", "risk_tier": 1, "owned_paths": [OWNED],
        "workspace_kind": "worktree", "workspace_path": str(build_repo),
    })
    assert created["ok"] is True, created
    with closing(kb.connect(board=SLUG)) as conn:
        card = kb.get_task(conn, created["task_id"])
        run_id, timed_out, signalled = _build_past_its_box(conn, card.id, saved)
        task = kb.get_task(conn, card.id)
        kinds = [event.kind for event in kb.list_events(conn, card.id)]
        run = kb.get_run(conn, run_id)

    assert timed_out == []
    # Parked with the independent reviewer, not re-queued for a rebuild.
    assert (task.status, task.assignee) == ("review", "raphael-verifier")
    assert {"run_handover_completed", "review_requested"} <= set(kinds)
    # A lone saved patch goes over through the existing saved-result handover,
    # whose review park records its receipt.
    assert ("saved_result_handed_over" in kinds) == (REPORT not in saved)
    assert "timed_out" not in kinds
    assert run.outcome != "timed_out"
    assert signalled == [WORKER_PID]
    # The saved patch is the head under review, materialized in the worktree.
    saved_file = Path(task.workspace_path) / OWNED / "saved.py"
    assert saved_file.read_text(encoding="utf-8") == "saved = True\n"
    # Decision 4 is the kernel's: every new governed build card is reviewed.
    assert (card.requires_review, card.max_runtime_seconds) == (True, BUILD_BOX)


def test_a_review_run_gets_the_review_box(build_repo):
    """Decision 2: a review run gets 45 minutes; its card keeps two hours."""
    with closing(kb.connect(board=SLUG)) as conn:
        task_id = _locked_card(
            conn, "raphael-claude-worker", "deep", risk_tier=1, owned_paths=[OWNED],
            workspace_path=str(build_repo), requires_review=True,
        )
        _build_past_its_box(conn, task_id, (PATCH, REPORT))
        host = kb._claimer_id().split(":", 1)[0]
        review = kb.claim_review_task(conn, task_id, claimer=f"{host}:r0")
        timed_out, signalled = _outlast(
            conn, task_id, review.current_run_id, DEEP_BOX, WORKER_PID + 1,
        )
        boxes = [
            row["max_runtime_seconds"] for row in conn.execute(
                "SELECT max_runtime_seconds FROM task_runs WHERE task_id = ? ORDER BY id",
                (task_id,),
            )
        ]
        task = kb.get_task(conn, task_id)

    assert timed_out == [task_id]
    assert set(signalled) == {WORKER_PID + 1}
    # The build run kept the build box and the review run got the review box;
    # the card keeps its own box for an implementation handback.
    assert boxes == [BUILD_BOX, DEEP_BOX]
    assert (task.max_runtime_seconds, task.status) == (BUILD_BOX, "review")


# ---------------------------------------------------------------------------
# kanban_create carries risk_tier, under the worker's own environment
# ---------------------------------------------------------------------------

WORKER_BOARD = "worker-board"
OTHER_BOARD = "other-board"


def _count(db: Path, where: str = "1 = 1", params: tuple = ()) -> int:
    """An independent read: a plain sqlite connection, no Hermes code."""
    if not db.exists():
        return 0
    with closing(sqlite3.connect(db)) as raw:
        return raw.execute(
            f"SELECT COUNT(*) FROM tasks WHERE {where}", params,
        ).fetchone()[0]


def _raw_row(db: Path, task_id: str) -> sqlite3.Row:
    with closing(sqlite3.connect(db)) as raw:
        raw.row_factory = sqlite3.Row
        return raw.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()


def test_kanban_create_carries_the_tier_under_the_worker_environment(
    fence_home, monkeypatch,
):
    from tools import kanban_tools

    root = fence_home
    kb.create_board(WORKER_BOARD)
    kb.write_board_metadata(WORKER_BOARD, project_id="p_governed_test")
    kb.create_board(OTHER_BOARD)
    with closing(kb.connect(board=OTHER_BOARD)) as conn:
        kb.create_task(conn, title="Someone else's card", assignee="default")
    with closing(kb.connect(board=WORKER_BOARD)) as conn:
        coordinator = _locked_card(conn, "raphael-planner", "deep", risk_tier=1, owned_paths=[])
    worker_db = root / "kanban" / "boards" / WORKER_BOARD / "kanban.db"
    other_db = root / "kanban" / "boards" / OTHER_BOARD / "kanban.db"
    default_db = root / "kanban.db"
    register = kb.register_db_path()
    workspaces = kb.workspaces_root(board=WORKER_BOARD)
    assert worker_db.exists() and other_db.exists()
    before = {db: _count(db) for db in (worker_db, other_db, default_db)}

    # Exactly what the dispatcher injects into the worker it spawns.
    profile_home = root / "profiles" / "raphael-planner"
    profile_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    monkeypatch.setenv("HERMES_PROFILE", "raphael-planner")
    monkeypatch.setenv("HERMES_KANBAN_DB", str(kb.kanban_db_path(board=WORKER_BOARD)))
    monkeypatch.setenv("HERMES_KANBAN_BOARD", WORKER_BOARD)
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACES_ROOT", str(workspaces))
    monkeypatch.setenv("HERMES_KANBAN_TASK", coordinator)

    # From the profile home the worker still reaches the root: its board
    # registry, every board's directory and the governed board's owner.
    assert kb.kanban_home().resolve() == root.resolve()
    assert kb.register_db_path() == register
    assert kb.board_dir(OTHER_BOARD).resolve() == other_db.parent.resolve()
    assert kb._board_owner_project_id(WORKER_BOARD) == "p_governed_test"

    created = json.loads(kanban_tools._handle_create({
        "title": "Decide the rollout", "assignee": "raphael-business",
        "execution_tier": "routine", "owned_paths": [], "risk_tier": 2,
    }))
    routine = json.loads(kanban_tools._handle_create({
        "title": "Draft the release note", "assignee": "raphael-business",
        "execution_tier": "routine", "owned_paths": [], "risk_tier": 0,
    }))
    refused = [
        json.loads(kanban_tools._handle_create({
            "title": f"Bad tier {label}", "assignee": "raphael-business",
            "execution_tier": "routine", "owned_paths": [], "risk_tier": value,
        }))
        for label, value in (("3", 3), ("true", True), ("string", "1"), ("minus", -1))
    ]

    assert created["ok"] is True and routine["ok"] is True
    # The tool's own proof ...
    assert (
        created["risk_tier"], created["reasoning_effort"], created["max_runtime_seconds"],
    ) == (2, "max", ROUTINE_BOX)
    assert (
        routine["risk_tier"], routine["reasoning_effort"], routine["max_runtime_seconds"],
    ) == (0, "high", ROUTINE_BOX)
    for answer in refused:
        assert "ok" not in answer or answer["ok"] is not True
        assert "risk_tier" in json.dumps(answer)
    # ... against an independent count of the rows each database holds.
    assert _count(worker_db) == before[worker_db] + 2
    assert _count(other_db) == before[other_db]
    assert _count(default_db) == before[default_db]
    assert not (profile_home / "kanban.db").exists()
    assert not (profile_home / "kanban").exists()
    # Decision 5: tier-2 work on the high-base lane (business routine on
    # Sonnet 5.5) runs at max; tier 0 stays high. Both seals read back clean.
    assert _count(
        worker_db,
        "risk_tier = 2 AND reasoning_effort = 'max' AND model_override = ? "
        "AND max_runtime_seconds = ? AND id = ?",
        ("claude-sonnet-5-5", ROUTINE_BOX, created["task_id"]),
    ) == 1
    assert _count(
        worker_db,
        "risk_tier = 0 AND reasoning_effort = 'high' AND model_override = ? "
        "AND max_runtime_seconds = ? AND id = ?",
        ("claude-sonnet-5-5", ROUTINE_BOX, routine["task_id"]),
    ) == 1
    for answer in (created, routine):
        assert kb.task_policy_lock_error(_raw_row(worker_db, answer["task_id"])) is None


def test_a_new_card_without_a_tier_is_pinned_as_tier_2(fence_home, monkeypatch):
    """Decision 3: a new governed card without a tier counts as tier 2, so max."""
    kb.create_board(WORKER_BOARD, project_id="p_governed_test")
    monkeypatch.setenv("HERMES_KANBAN_BOARD", WORKER_BOARD)

    created = _kanban_create({
        "title": "Read the plan", "assignee": "raphael-business",
        "execution_tier": "routine", "owned_paths": [],
    })

    assert created["ok"] is True, created
    row = _raw_row(kb.kanban_db_path(board=WORKER_BOARD), created["task_id"])
    # Business routine is the one lane whose own effort is high (Sonnet 5.5).
    assert (row["risk_tier"], row["model_override"]) == (2, "claude-sonnet-5-5")
    assert (row["reasoning_effort"], row["max_runtime_seconds"]) == ("max", ROUTINE_BOX)
    assert kb.task_policy_lock_error(row) is None


def test_kanban_create_refuses_a_tier_on_a_board_no_owner_governs(kanban_home):
    from tools import kanban_tools

    db = kanban_home / "kanban.db"
    before = _count(db)

    answer = json.loads(kanban_tools._handle_create({
        "title": "Ungoverned work", "assignee": "default", "risk_tier": 1,
    }))

    assert answer.get("ok") is not True
    assert "risk_tier" in json.dumps(answer)
    assert _count(db) == before


# ---------------------------------------------------------------------------
# A new card records the tier it counts as; a card from before does not
# ---------------------------------------------------------------------------

# A box no kind of work is pinned at, as a card from before may hold.
LEGACY_BOX = 3600


def _review_round_trip(conn, task_id: str) -> list:
    """Creation, review and a typed non-empty findings handback: the row after each."""
    rows = [_row(conn, task_id)]
    _build_past_its_box(conn, task_id, (PATCH,))
    rows.append(_row(conn, task_id))
    host = kb._claimer_id().split(":", 1)[0]
    review = kb.claim_review_task(conn, task_id, claimer=f"{host}:r0")
    handback = kb.submit_review_findings(
        conn, task_id, findings=[_finding()], candidate_digest="digest-1",
        expected_run_id=review.current_run_id,
    )
    assert handback["outcome"] == "handed_back", handback
    rows.append(_row(conn, task_id))
    assert [(row["status"], row["assignee"]) for row in rows[1:]] == [
        ("review", "raphael-verifier"), ("ready", rows[0]["assignee"]),
    ]
    for row in rows:
        assert kb.task_policy_lock_error(row) is None
    return rows


@pytest.mark.parametrize(
    ("risk_tier", "recorded", "effort"), [(None, 2, "max"), (1, 1, "high")],
    ids=["omitted-tier", "tier-1"],
)
def test_kanban_create_keeps_the_tier_effort_through_review(
    build_repo, risk_tier, recorded, effort,
):
    """An omitted tier is recorded as tier 2 and keeps max; tier 1 keeps high."""
    # Business routine is the one lane whose own effort is high (Sonnet 5.5).
    args = {
        "title": "Write the pricing page", "assignee": "raphael-business",
        "execution_tier": "routine", "owned_paths": [OWNED],
        "workspace_kind": "worktree", "workspace_path": str(build_repo),
    }
    if risk_tier is not None:
        args["risk_tier"] = risk_tier
    created = _kanban_create(args)
    assert created["ok"] is True, created
    with closing(kb.connect(board=SLUG)) as conn:
        rows = _review_round_trip(conn, created["task_id"])

    assert [(row["risk_tier"], row["reasoning_effort"]) for row in rows] == [
        (recorded, effort),
    ] * 3
    assert created["risk_tier"] == recorded
    assert rows[-1]["model_override"] == "claude-sonnet-5-5"


def test_a_native_security_review_card_keeps_max_through_review(build_repo):
    """The same for a card made through create_task as a security review."""
    with closing(kb.connect(board=SLUG)) as conn:
        task_id = _locked_card(
            conn, "raphael-business", "routine", risk_tier=None,
            responsibility="R12", owned_paths=[OWNED], workspace_path=str(build_repo),
        )
        rows = _review_round_trip(conn, task_id)

    assert [(row["risk_tier"], row["reasoning_effort"]) for row in rows] == [
        (2, "max"),
    ] * 3
    assert rows[-1]["model_override"] == "claude-sonnet-5-5"


def test_a_legacy_card_keeps_its_base_effort_and_saved_box_through_review(build_repo):
    """Decision 3: a card from before the tier is not repinned."""
    route = mp.task_assignment_for("raphael-business", "anthropic", "routine")
    with closing(kb.connect(board=SLUG)) as conn:
        task_id = _locked_card(
            conn, "raphael-business", "routine", risk_tier=None,
            owned_paths=[OWNED], workspace_path=str(build_repo),
        )
        # The row as it was written before: no tier, the lane's own effort
        # sealed, and the box it was given then.
        conn.execute(
            "UPDATE tasks SET risk_tier = NULL, reasoning_effort = ?, "
            "model_policy_lock = ?, max_runtime_seconds = ? WHERE id = ?",
            (
                route.reasoning_effort,
                kb.mint_policy_lock(
                    "raphael-business", route.provider, route.model,
                    route.reasoning_effort, "routine",
                ),
                LEGACY_BOX, task_id,
            ),
        )
        _as_written_before_the_pins(conn, task_id)
        rows = _review_round_trip(conn, task_id)

    # Each role's own effort: high for business, max for the reviewer.
    assert [row["reasoning_effort"] for row in rows] == ["high", "max", "high"]
    assert [(row["risk_tier"], row["max_runtime_seconds"]) for row in rows] == [
        (None, LEGACY_BOX),
    ] * 3


def _as_written_before_the_pins(conn, task_id):
    """Clear the creation-pin record that rows written before it never got."""
    columns = {row[1] for row in conn.execute("PRAGMA table_info(tasks)")}
    if "pinned_effort" in columns:
        conn.execute("UPDATE tasks SET pinned_effort = NULL WHERE id = ?", (task_id,))


@pytest.mark.parametrize(
    ("assignee", "lane", "risk_tier", "responsibility", "efforts"),
    [
        ("raphael-business", "routine", 2, None, ["high", "max", "high"]),
        ("raphael-business", "routine", 0, "R12", ["high", "max", "high"]),
        ("raphael-claude-worker", "deep", 1, None, ["max", "max", "max"]),
    ],
    ids=["tier-2-business", "tier-0-security-review", "tier-1-build"],
)
def test_a_card_with_a_tier_from_before_the_pins_keeps_its_base_effort_through_review(
    build_repo, assignee, lane, risk_tier, responsibility, efforts,
):
    """A tier recorded before the creation-time pins is not proof of a pin."""
    route = mp.task_assignment_for(assignee, "anthropic", lane)
    with closing(kb.connect(board=SLUG)) as conn:
        task_id = _locked_card(
            conn, assignee, lane, risk_tier=None, responsibility=responsibility,
            owned_paths=[OWNED], workspace_path=str(build_repo),
        )
        # The row as the earlier tier-carrying kernel wrote it: a tier, the
        # lane's own effort sealed and the box it was given then.
        conn.execute(
            "UPDATE tasks SET risk_tier = ?, reasoning_effort = ?, "
            "model_policy_lock = ?, max_runtime_seconds = ? WHERE id = ?",
            (
                risk_tier,
                route.reasoning_effort,
                kb.mint_policy_lock(
                    assignee, route.provider, route.model,
                    route.reasoning_effort, lane,
                ),
                LEGACY_BOX, task_id,
            ),
        )
        _as_written_before_the_pins(conn, task_id)
        rows = _review_round_trip(conn, task_id)

    # Each role's own effort, as before the pins.
    assert [row["reasoning_effort"] for row in rows] == efforts
    assert [(row["risk_tier"], row["max_runtime_seconds"]) for row in rows] == [
        (risk_tier, LEGACY_BOX),
    ] * 3
