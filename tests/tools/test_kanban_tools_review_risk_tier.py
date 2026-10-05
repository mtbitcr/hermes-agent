"""A reviewer may raise a card's risk tier through ``kanban_review_findings``.

Tier plan section d, P3-1 to P3-4 (section j, decision 1). Every call is
dispatched through the REAL tool registry, as in
``test_kanban_tools_review_findings.py``, on real boards and a real git
worktree, under the environment the dispatcher injects into the reviewer:
``HERMES_KANBAN_DB``/``HERMES_KANBAN_BOARD`` pinned to the worker's own
board, ``HERMES_KANBAN_TASK``/``HERMES_KANBAN_RUN_ID`` set, and
``HERMES_HOME`` at a profile home under the root.
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
from contextlib import closing
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from plugins.dashboard_auth.raphael_workspace import model_policy as mp

WORKER_BOARD = "worker-board"
OTHER_BOARD = "other-board"
IMPLEMENTER = "raphael-business"


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        [
            "git", "-C", str(cwd),
            "-c", "user.name=Test User",
            "-c", "user.email=test@example.com",
            "-c", "commit.gpgsign=false",
            *args,
        ],
        check=True, capture_output=True, text=True,
    ).stdout.strip()


def _finding(head: str) -> dict:
    return {
        "severity": "major",
        "file": "src/app.py",
        "lines": "1",
        "problem": "the change touches the payment path",
        "impact": "a wrong total reaches the customer",
        "smallest_fix": "check the total before it is sent",
        "candidate_digest": head,
    }


def _count(db: Path, table: str, where: str, params: tuple = ()) -> int:
    """An independent read: a plain sqlite connection, no Hermes code."""
    if not db.exists():
        return 0
    with closing(sqlite3.connect(db)) as raw:
        return raw.execute(
            f"SELECT COUNT(*) FROM {table} WHERE {where}", params,
        ).fetchone()[0]


def _events(conn, tid: str, kind: str) -> list:
    return [
        json.loads(row["payload"])
        for row in conn.execute(
            "SELECT payload FROM task_events WHERE task_id = ? AND kind = ? ORDER BY id",
            (tid, kind),
        )
    ]


def _snapshot(conn, tid: str) -> tuple:
    """Everything a refused call must leave as it was."""
    row = conn.execute(
        "SELECT status, assignee, risk_tier, reasoning_effort, model_policy_lock, "
        "current_run_id FROM tasks WHERE id = ?", (tid,),
    ).fetchone()
    events = conn.execute(
        "SELECT id, kind FROM task_events WHERE task_id = ? ORDER BY id", (tid,),
    ).fetchall()
    attachments = conn.execute(
        "SELECT COUNT(*) FROM task_attachments WHERE task_id = ?", (tid,),
    ).fetchone()[0]
    return tuple(row), [tuple(event) for event in events], attachments


def _dispatch(args: dict) -> dict:
    from tools.registry import registry
    import tools.kanban_tools  # noqa: F401 - ensure registered

    entry = registry.get_entry("kanban_review_findings")
    assert entry is not None and entry.toolset == "kanban"
    return json.loads(entry.handler(args))


@pytest.fixture
def reviewing(tmp_path, monkeypatch):
    """Park a tier-``n`` card in review and pin the reviewer's own environment."""
    root = tmp_path / "hermes_home"
    root.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    for var in (
        "HERMES_KANBAN_DB", "HERMES_KANBAN_WORKSPACES_ROOT", "HERMES_KANBAN_HOME",
        "HERMES_KANBAN_BOARD", "HERMES_KANBAN_TASK", "HERMES_KANBAN_RUN_ID",
        "HERMES_SESSION_ID", "HERMES_PROFILE",
    ):
        monkeypatch.delenv(var, raising=False)
    kb._INITIALIZED_PATHS.clear()
    kb._REGISTER_INITIALIZED = False
    kb._REGISTER_INITIALIZED_PATHS.clear()
    kb._ARCHIVE_INITIALIZED_PATHS.clear()
    import hermes_constants

    hermes_constants._cached_default_hermes_root = None
    # Stand in for each role's on-disk config only, as test_owner_workspace
    # does; the policy, the pins and the seals run as production code.
    monkeypatch.setattr(
        mp, "configured_assignment_for",
        lambda profile: mp.assignment_for(profile, "anthropic"),
    )
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "init")
    kb.create_board(WORKER_BOARD)
    kb.create_board(OTHER_BOARD)
    with closing(kb.connect(board=OTHER_BOARD)) as other:
        kb.create_task(other, title="Someone else's card", assignee="default")
    monkeypatch.setenv("HERMES_KANBAN_BOARD", WORKER_BOARD)
    host = kb._claimer_id().split(":", 1)[0]
    conns = []

    def park(tier: int, *, pinned: bool = True):
        conn = kb.connect(board=WORKER_BOARD)
        conns.append(conn)
        route = mp.task_assignment_for(IMPLEMENTER, "anthropic", "routine")
        tid = kb.create_task(
            conn, title=f"Payment change at tier {tier}", assignee=IMPLEMENTER,
            execution_tier="routine", provider_override=route.provider,
            model_override=route.model, reasoning_effort=route.reasoning_effort,
            model_policy_lock=kb.mint_policy_lock(
                IMPLEMENTER, route.provider, route.model, route.reasoning_effort,
                "routine",
            ),
            owned_paths=["src"], workspace_kind="worktree",
            workspace_path=str(repo), requires_review=True, risk_tier=tier,
        )
        if not pinned:
            # A card written before the creation pins: no pinned_effort.
            conn.execute("UPDATE tasks SET pinned_effort = NULL WHERE id = ?", (tid,))
        run = kb.claim_task(conn, tid, claimer=f"{host}:w0")
        assert run is not None
        workspace, branch = kb._resolve_worktree_workspace(run)
        kb.set_workspace_path(conn, tid, workspace)
        kb.set_branch_name(conn, tid, branch)
        kb.record_worktree_base(conn, tid, workspace)
        (workspace / "src").mkdir(parents=True, exist_ok=True)
        (workspace / "src" / "app.py").write_text("total = 1\n", encoding="utf-8")
        _git(workspace, "add", "src/app.py")
        _git(workspace, "commit", "-m", "feat: payment change")
        head = _git(workspace, "rev-parse", "HEAD")
        assert kb.request_review(
            conn, tid, summary="ready", reviewer=kb.policy_resolved_reviewer(),
            expected_run_id=run.current_run_id,
        )
        review = kb.claim_review_task(conn, tid, claimer=f"{host}:r0")
        assert review is not None
        # Exactly what the dispatcher injects into the reviewer it spawns.
        worker_db = kb.kanban_db_path(board=WORKER_BOARD)
        workspaces = kb.workspaces_root(board=WORKER_BOARD)
        profile_home = root / "profiles" / review.assignee
        profile_home.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("HERMES_HOME", str(profile_home))
        monkeypatch.setenv("HERMES_PROFILE", review.assignee)
        monkeypatch.setenv("HERMES_KANBAN_DB", str(worker_db))
        monkeypatch.setenv("HERMES_KANBAN_WORKSPACES_ROOT", str(workspaces))
        monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
        monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(review.current_run_id))
        return conn, tid, head, review

    park.root = root
    park.host = host
    park.register = kb.register_db_path()
    yield park
    for conn in conns:
        conn.close()


# ---------------------------------------------------------------------------
# P3-1: a raise is accepted, once
# ---------------------------------------------------------------------------


def test_a_reviewer_raise_is_accepted(reviewing):
    conn, tid, head, review = reviewing(1)
    root = reviewing.root
    worker_db = root / "kanban" / "boards" / WORKER_BOARD / "kanban.db"
    other_db = root / "kanban" / "boards" / OTHER_BOARD / "kanban.db"
    default_db = root / "kanban.db"
    profile_home = root / "profiles" / review.assignee
    # From the profile home the reviewer still reaches the root: its board
    # registry and every other board.
    assert kb.kanban_home().resolve() == root.resolve()
    assert kb.register_db_path() == reviewing.register
    assert kb.board_dir(OTHER_BOARD).resolve() == other_db.parent.resolve()
    before_other = _count(other_db, "tasks", "1 = 1")
    before_default = _count(default_db, "tasks", "1 = 1")

    out = _dispatch({"findings": [_finding(head)], "risk_tier": 2})

    raised = {"from": 1, "to": 2, "reviewer": review.assignee, "run_id": review.current_run_id}
    # The tool's own proof ...
    assert out.get("ok") is True, out
    assert out["outcome"] == "handed_back"
    assert out["risk_tier_raised"] == raised
    assert kb.get_task(conn, tid).risk_tier == 2
    assert _events(conn, tid, "risk_tier_raised") == [raised]
    # ... against an independent count of what each database holds.
    assert _count(worker_db, "tasks", "id = ? AND risk_tier = 2", (tid,)) == 1
    assert _count(
        worker_db, "task_events", "task_id = ? AND kind = 'risk_tier_raised' AND run_id = ?",
        (tid, review.current_run_id),
    ) == 1
    assert _count(worker_db, "task_events", "kind = 'risk_tier_raised'") == 1
    assert _count(other_db, "tasks", "1 = 1") == before_other
    assert _count(other_db, "task_events", "kind = 'risk_tier_raised'") == 0
    assert _count(default_db, "tasks", "1 = 1") == before_default
    assert not (profile_home / "kanban.db").exists()
    assert not (profile_home / "kanban").exists()


# ---------------------------------------------------------------------------
# P3-2: a lowering is refused by the kernel, with no write
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("tier", "lower"), [(2, 1), (1, 0)], ids=["2-to-1", "1-to-0"])
def test_a_reviewer_lowering_is_refused_with_no_write(reviewing, tier, lower):
    conn, tid, head, review = reviewing(tier)
    before = _snapshot(conn, tid)

    out = _dispatch({"findings": [_finding(head)], "risk_tier": lower})

    assert out.get("ok") is not True, out
    assert "risk_tier" in out["error"]
    assert _snapshot(conn, tid) == before
    assert before[0][:3] == ("running", review.assignee, tier)
    assert _events(conn, tid, "risk_tier_raised") == []
    assert _events(conn, tid, "review_findings_delivered") == []
    assert kb.list_attachments(conn, tid) == []
    # The kernel itself refuses it, whatever the caller.
    refused = kb.submit_review_findings(
        conn, tid, findings=[_finding(head)], candidate_digest=head,
        expected_run_id=review.current_run_id, risk_tier=lower,
    )
    assert refused["outcome"] == "error" and "risk_tier" in refused["reason"]
    assert _snapshot(conn, tid) == before


# ---------------------------------------------------------------------------
# P3-3: only the active review run can raise
# ---------------------------------------------------------------------------


def test_only_the_active_review_run_can_raise(reviewing, monkeypatch):
    from agent.delegation_context import delegated_child_context

    conn, tid, head, review = reviewing(0)
    raise_to_1 = {"findings": [_finding(head)], "risk_tier": 1}

    before = _snapshot(conn, tid)
    monkeypatch.delenv("HERMES_KANBAN_RUN_ID")
    missing = _dispatch(raise_to_1)
    assert missing.get("ok") is not True and "active reviewer run" in missing["error"]
    assert _snapshot(conn, tid) == before
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(review.current_run_id))
    with delegated_child_context():
        child = _dispatch(raise_to_1)
    assert child.get("ok") is not True and "delegate_task child" in child["error"]
    assert _snapshot(conn, tid) == before

    # The active review run can.
    handed_back = _dispatch(raise_to_1)
    assert handed_back.get("ok") is True, handed_back
    assert kb.get_task(conn, tid).risk_tier == 1

    # The implementer's next run cannot.
    run = kb.claim_task(conn, tid, claimer=f"{reviewing.host}:w1")
    assert run is not None
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run.current_run_id))
    before = _snapshot(conn, tid)
    implementer = _dispatch({"findings": [_finding(head)], "risk_tier": 2})
    assert implementer.get("ok") is not True, implementer
    assert _snapshot(conn, tid) == before
    assert [e["to"] for e in _events(conn, tid, "risk_tier_raised")] == [1]
    assert kb.get_task(conn, tid).risk_tier == 1


# ---------------------------------------------------------------------------
# P3-4: a raise to tier 2 re-pins effort to max at the handback
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("pinned", "effort"), [(True, "max"), (False, "high")],
    ids=["pinned-card", "card-from-before-the-pins"],
)
def test_a_raise_to_tier_2_repins_effort_at_handback(reviewing, pinned, effort):
    """Decision 1; a card without pinned_effort keeps the role's base effort."""
    conn, tid, head, review = reviewing(1, pinned=pinned)

    out =_dispatch({"findings": [_finding(head)], "risk_tier": 2})

    assert out.get("ok") is True, out
    assert (out["outcome"], out["implementer"]) == ("handed_back", IMPLEMENTER)
    row = conn.execute("SELECT * FROM tasks WHERE id = ?", (tid,)).fetchone()
    assert (row["status"], row["assignee"], row["risk_tier"]) == ("ready", IMPLEMENTER, 2)
    assert (row["model_override"], row["reasoning_effort"]) == ("claude-sonnet-5", effort)
    assert kb.task_policy_lock_error(row) is None
    # The implementer's next run starts on that seal.
    run = kb.claim_task(conn, tid, claimer=f"{reviewing.host}:w1")
    assert run is not None
    row = conn.execute("SELECT * FROM tasks WHERE id = ?", (tid,)).fetchone()
    assert (row["status"], row["reasoning_effort"]) == ("running", effort)
    assert kb.task_policy_lock_error(row) is None
