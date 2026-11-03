"""A verdict through the real kanban_review_findings tool on a card that changes no file."""

from __future__ import annotations

import json
import os
import re
import sqlite3
import subprocess
from pathlib import Path

import pytest

# The new rule's module is not imported here, so old code fails on behaviour, not an import.
REVIEWER = "raphael-verifier"  # a profile the model policy admits as reviewer
IMPLEMENTER = "test-worker"
NO_HEAD_REFUSAL = "no repository head was parked"


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        [
            "git", "-C", str(cwd),
            "-c", "user.name=Test User",
            "-c", "user.email=test@example.com",
            "-c", "commit.gpgsign=false",
            *args,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _finding(candidate_digest: str) -> dict:
    return {
        "severity": "major",
        "file": "linkedin-post.txt",
        "lines": "1-2",
        "problem": "the post names the wrong product",
        "impact": "the published text misstates what the owner approved",
        "smallest_fix": "correct the product name in the post and republish",
        "candidate_digest": candidate_digest,
    }


def _dispatch(args: dict) -> dict:
    """Look ``kanban_review_findings`` up in the real registry and call it."""
    from tools.registry import registry

    entry = registry.get_entry("kanban_review_findings")
    assert entry is not None, "kanban_review_findings is not registered"
    return json.loads(entry.handler(args))


def _write_config(home: Path, *, enabled: bool) -> None:
    (home / "config.yaml").write_text(
        "kanban:\n"
        "  delivery:\n"
        f"    enabled: {'true' if enabled else 'false'}\n",
        encoding="utf-8",
    )


# Reading the board file directly, past the kernel and the tool's return.


def _sql(db_path: Path, statement: str, *params):
    raw = sqlite3.connect(str(db_path))
    try:
        return raw.execute(statement, params).fetchall()
    finally:
        raw.close()


def _count(db_path: Path, statement: str, *params) -> int:
    return _sql(db_path, statement, *params)[0][0]


def _state(db_path: Path, tid: str) -> dict:
    """What the board file itself holds for the card and for deliveries."""
    status, assignee = _sql(
        db_path, "SELECT status, assignee FROM tasks WHERE id = ?", tid,
    )[0]
    return {
        "status": status,
        "assignee": assignee,
        "approvals": _count(
            db_path,
            "SELECT COUNT(*) FROM task_events WHERE task_id = ? AND kind = 'completed'",
            tid,
        ),
        "handbacks": _count(
            db_path,
            "SELECT COUNT(*) FROM task_events "
            "WHERE task_id = ? AND kind = 'review_findings_delivered'",
            tid,
        ),
        "attachments": _count(
            db_path, "SELECT COUNT(*) FROM task_attachments WHERE task_id = ?", tid,
        ),
        "deliveries": _count(db_path, "SELECT COUNT(*) FROM kanban_deliveries"),
    }


def _release_entries(root: Path) -> int:
    """Rows in every release table of every release store under the root."""
    total = 0
    for ledger in root.rglob("release_ledger.db"):
        tables = _sql(
            ledger,
            "SELECT name FROM sqlite_master "
            "WHERE type = 'table' AND name LIKE 'release\\_%' ESCAPE '\\'",
        )
        for (table,) in tables:
            total += _count(ledger, f'SELECT COUNT(*) FROM "{table}"')
    return total


def _rewrite_handover(db_path: Path, tid: str, *, drop: tuple[str, ...]) -> None:
    """Rewrite the newest handover record as a build that recorded less would have."""
    raw = sqlite3.connect(str(db_path))
    try:
        event_id, payload = raw.execute(
            "SELECT id, payload FROM task_events "
            "WHERE task_id = ? AND kind = 'review_requested' "
            "ORDER BY id DESC LIMIT 1",
            (tid,),
        ).fetchone()
        document = json.loads(payload)
        for key in drop:
            del document[key]
        raw.execute(
            "UPDATE task_events SET payload = ? WHERE id = ?",
            (json.dumps(document), event_id),
        )
        raw.commit()
    finally:
        raw.close()


# Fixtures: a root, a card parked for review, the worker's pins.


@pytest.fixture
def kanban(tmp_path, monkeypatch):
    """A Hermes root with an initialised default board: ``(kb, root)``."""
    root = tmp_path / ".hermes"
    root.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("HERMES_SESSION_ID", raising=False)

    from hermes_cli import kanban_db as kb
    from plugins.dashboard_auth.raphael_workspace import model_policy
    import tools.kanban_tools  # noqa: F401 - registers the tool

    # The reviewer the model policy nominates: the seam the other review tests pin.
    monkeypatch.setattr(
        model_policy, "reviewer_profile_ids", lambda: (REVIEWER,), raising=True,
    )
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return kb, root


@pytest.fixture
def repo(tmp_path) -> Path:
    origin = tmp_path / "repo"
    origin.mkdir()
    subprocess.run(
        ["git", "init", "-b", "main", str(origin)],
        check=True, capture_output=True, text=True,
    )
    (origin / "README.md").write_text("base\n", encoding="utf-8")
    _git(origin, "add", "README.md")
    _git(origin, "commit", "-m", "init")
    return origin


def _hand_over(kb, conn, *, owned_paths, requires_review: bool = False) -> str:
    """A card worked by its implementer and parked for review by the kernel."""
    tid = kb.create_task(
        conn,
        title="publish the approved post",
        assignee=IMPLEMENTER,
        owned_paths=owned_paths,
        requires_review=requires_review,
    )
    run = kb.claim_task(conn, tid, claimer=f"{IMPLEMENTER}:1")
    assert run is not None
    parked, why = kb.request_review(
        conn, tid, summary="published; the post link is on the card",
        reviewer=REVIEWER, expected_run_id=run.current_run_id, with_reason=True,
    )
    assert parked, f"the kernel refused the review handover: {why}"
    return tid


def _pin_worker(monkeypatch, db_path: Path, board: str, tid: str, run_id) -> None:
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    monkeypatch.setenv("HERMES_KANBAN_BOARD", board)
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))


def _review_card(kb, monkeypatch, *, owned_paths, requires_review: bool = False):
    """A card parked for review, claimed from the review lane, pinned as its worker."""
    db_path = kb.kanban_db_path()
    conn = kb.connect()
    try:
        tid = _hand_over(
            kb, conn, owned_paths=owned_paths, requires_review=requires_review,
        )
        review = kb.claim_review_task(conn, tid, claimer=f"{REVIEWER}:1")
        assert review is not None
    finally:
        conn.close()
    _pin_worker(monkeypatch, db_path, kb.DEFAULT_BOARD, tid, review.current_run_id)
    return tid, db_path


def _record_digest(db_path: Path, tid: str) -> str:
    """The bound digest, as a reviewer learns it: from the refusal of a wrong finding digest."""
    out = _dispatch({"findings": [_finding("probe")]})
    match = re.search(r"document candidate_digest \('([^']+)'\)", out.get("error") or "")
    assert match, f"the tool bound no candidate for {tid}: {out}"
    assert _state(db_path, tid)["status"] == "running", "the probe wrote to the card"
    return match.group(1)


# (a) an empty verdict approves a card without code and delivers nothing


@pytest.mark.parametrize("delivery_on", [False, True], ids=["delivery-off", "delivery-on"])
@pytest.mark.parametrize("owned_paths", [[], None], ids=["read-only-scope", "no-scope"])
def test_empty_verdict_approves_a_card_without_code_and_delivers_nothing(
    kanban, monkeypatch, owned_paths, delivery_on,
):
    kb, root = kanban
    if delivery_on:
        _write_config(root, enabled=True)
    tid, db_path = _review_card(
        kb, monkeypatch, owned_paths=owned_paths, requires_review=delivery_on,
    )
    if delivery_on:
        # Review required and delivery on: only the missing head blocks a delivery.
        from hermes_cli import kanban_delivery

        assert kanban_delivery.delivery_settings() is True
        assert _count(
            db_path, "SELECT requires_review FROM tasks WHERE id = ?", tid,
        ) == 1
    before = _state(db_path, tid)
    assert before["status"] == "running" and before["approvals"] == 0

    out = _dispatch({"findings": []})
    assert out.get("ok") is True, out
    assert out["outcome"] == "passed"

    after = _state(db_path, tid)
    assert after["status"] == "done"
    assert after["approvals"] == 1
    assert after["deliveries"] == 0
    assert _release_entries(root) == 0


# (b) findings return the card to its assignee


def test_findings_on_a_card_without_code_return_it_to_its_assignee(kanban, monkeypatch):
    kb, root = kanban
    tid, db_path = _review_card(kb, monkeypatch, owned_paths=[])
    digest = _record_digest(db_path, tid)

    out = _dispatch({"findings": [_finding(digest)]})
    assert out.get("ok") is True, out
    assert out["outcome"] == "handed_back"
    assert out["implementer"] == IMPLEMENTER

    after = _state(db_path, tid)
    assert (after["status"], after["assignee"]) == ("ready", IMPLEMENTER)
    assert after["handbacks"] == 1
    assert after["attachments"] == 1
    assert after["approvals"] == 0
    assert after["deliveries"] == 0
    assert _release_entries(root) == 0


# (c) a candidate the caller names is refused when it differs


@pytest.mark.parametrize("then_supplying", ["nothing", "the-record-candidate"])
def test_a_different_caller_candidate_is_refused_and_the_card_still_approves(
    kanban, monkeypatch, then_supplying,
):
    kb, root = kanban
    tid, db_path = _review_card(kb, monkeypatch, owned_paths=[])

    wrong = _dispatch({"findings": [], "candidate_digest": "0" * 40})
    assert wrong.get("ok") is not True, wrong
    assert "candidate_digest" in wrong["error"], wrong
    assert NO_HEAD_REFUSAL not in wrong["error"], wrong
    refused = _state(db_path, tid)
    assert (refused["status"], refused["approvals"], refused["deliveries"]) == (
        "running", 0, 0,
    )

    # The same card, the same run: only the candidate argument differs.
    args: dict = {"findings": []}
    if then_supplying == "the-record-candidate":
        args["candidate_digest"] = _record_digest(db_path, tid)
    approved = _dispatch(args)
    assert approved.get("ok") is True, approved
    assert approved["outcome"] == "passed"
    assert _state(db_path, tid)["status"] == "done"


def test_the_record_candidate_of_an_older_handover_is_refused_after_a_second_handover(
    kanban, monkeypatch,
):
    kb, root = kanban
    tid, db_path = _review_card(kb, monkeypatch, owned_paths=[])
    first = _record_digest(db_path, tid)
    handed_back = _dispatch({"findings": [_finding(first)]})
    assert handed_back.get("outcome") == "handed_back", handed_back

    # The implementer reworks and hands the card over again.
    conn = kb.connect()
    try:
        run = kb.claim_task(conn, tid, claimer=f"{IMPLEMENTER}:2")
        assert run is not None
        parked, why = kb.request_review(
            conn, tid, summary="reworked", reviewer=REVIEWER,
            expected_run_id=run.current_run_id, with_reason=True,
        )
        assert parked, f"the kernel refused the second handover: {why}"
        review = kb.claim_review_task(conn, tid, claimer=f"{REVIEWER}:2")
        assert review is not None
    finally:
        conn.close()
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(review.current_run_id))

    stale = _dispatch({"findings": [], "candidate_digest": first})
    assert stale.get("ok") is not True, stale
    assert "candidate_digest" in stale["error"], stale
    assert NO_HEAD_REFUSAL not in stale["error"], stale
    assert _state(db_path, tid)["status"] == "running"

    assert _record_digest(db_path, tid) != first
    approved = _dispatch({"findings": []})
    assert approved.get("ok") is True, approved
    assert approved["outcome"] == "passed"


# (d) a card that should carry code is still refused without a parked head


def test_a_card_with_owned_paths_and_no_parked_head_is_still_refused(
    kanban, repo, monkeypatch,
):
    """Passes on the old code too: it pins the refusal this change must keep."""
    kb, root = kanban
    db_path = kb.kanban_db_path()
    conn = kb.connect()
    try:
        tid = kb.create_task(
            conn,
            title="reviewed code",
            assignee=IMPLEMENTER,
            owned_paths=["src"],
            workspace_kind="worktree",
            workspace_path=str(repo),
            branch_name="feature/record-candidate",
        )
        run = kb.claim_task(conn, tid, claimer=f"{IMPLEMENTER}:1")
        assert run is not None
        workspace, branch = kb._resolve_worktree_workspace(run)
        kb.set_workspace_path(conn, tid, workspace)
        kb.set_branch_name(conn, tid, branch)
        kb.record_worktree_base(conn, tid, workspace)
        (workspace / "src").mkdir(parents=True, exist_ok=True)
        (workspace / "src" / "app.py").write_text("ok = True\n", encoding="utf-8")
        _git(workspace, "add", "src/app.py")
        _git(workspace, "commit", "-m", "feat: implement")
        parked, why = kb.request_review(
            conn, tid, summary="ready", reviewer=REVIEWER,
            expected_run_id=run.current_run_id, with_reason=True,
        )
        assert parked, f"the kernel refused the review handover: {why}"
        assert kb._latest_review_head_provenance(conn, tid), "the kernel parks a head"
        review = kb.claim_review_task(conn, tid, claimer=f"{REVIEWER}:1")
        assert review is not None
        # A handover recorded by a build that kept the scope but no head.
        _rewrite_handover(db_path, tid, drop=(kb._REVIEW_SCOPE_HEAD_KEY,))
        assert kb._latest_review_head_provenance(conn, tid) is None
        assert kb._latest_review_scope_provenance(conn, tid)["owned_paths"] == ["src"]
    finally:
        conn.close()
    _pin_worker(monkeypatch, db_path, kb.DEFAULT_BOARD, tid, review.current_run_id)

    for args in ({"findings": []}, {"findings": [], "candidate_digest": "0" * 40}):
        out = _dispatch(args)
        assert out.get("ok") is not True, out
        assert NO_HEAD_REFUSAL in out["error"], out
    after = _state(db_path, tid)
    assert (after["status"], after["approvals"], after["handbacks"]) == ("running", 0, 0)
    assert after["deliveries"] == 0


def test_a_handover_that_recorded_no_scope_is_still_refused(kanban, monkeypatch):
    """Nothing says the card changes no file, so the old refusal stands."""
    kb, root = kanban
    tid, db_path = _review_card(kb, monkeypatch, owned_paths=[])
    _rewrite_handover(
        db_path, tid,
        drop=(
            kb._REVIEW_SCOPE_PATHS_KEY,
            kb._REVIEW_SCOPE_INTEGRATES_KEY,
            kb._REVIEW_SCOPE_HEAD_KEY,
        ),
    )

    out = _dispatch({"findings": []})
    assert out.get("ok") is not True, out
    assert NO_HEAD_REFUSAL in out["error"], out
    after = _state(db_path, tid)
    assert (after["status"], after["approvals"]) == ("running", 0)


# (e) under the dispatcher's worker environment, on boards under the root


def _spawned_worker_env(kb, task, workspace: Path, board: str, monkeypatch) -> dict:
    """The env _default_spawn builds for a worker; only the process launch is stubbed."""
    launches = []

    class _Launched:
        pid = 4242

    def _launch(argv, **kwargs):
        log = kwargs.get("stdout")
        if hasattr(log, "close"):
            log.close()
        launches.append((list(argv), dict(kwargs.get("env") or {})))
        return _Launched()

    with monkeypatch.context() as patch:
        patch.setattr(subprocess, "Popen", _launch)
        pid = kb._default_spawn(task, str(workspace), board=board)
    assert pid == _Launched.pid
    assert len(launches) == 1, launches
    argv, env = launches[0]
    assert f"work kanban task {task.id}" in argv
    return env


def _become(env: dict, monkeypatch) -> None:
    """Carry on as the worker: this process now has exactly ``env``."""
    for name in list(os.environ):
        if name not in env:
            monkeypatch.delenv(name)
    for name, value in env.items():
        if os.environ.get(name) != value:
            monkeypatch.setenv(name, value)


def test_verdict_on_a_card_without_code_under_the_dispatcher_worker_environment(
    kanban, monkeypatch,
):
    kb, root = kanban
    profile_home = root / "profiles" / REVIEWER
    profile_home.mkdir(parents=True)
    _write_config(profile_home, enabled=True)

    workers = {}
    for slug in ("proj-a", "proj-b"):
        kb.create_board(slug)
        conn = kb.connect(board=slug)
        try:
            tid = _hand_over(kb, conn, owned_paths=[], requires_review=True)
            claimed = kb.claim_review_task(conn, tid, claimer=f"{REVIEWER}:1")
            assert claimed is not None and claimed.current_run_id is not None
            workspace = kb.resolve_workspace(claimed, board=slug)
            env = _spawned_worker_env(kb, claimed, workspace, slug, monkeypatch)
        finally:
            conn.close()
        db_path = root / "kanban" / "boards" / slug / "kanban.db"
        assert db_path.is_file()
        # Pinned to its own card, board and board file, living in a profile home.
        assert env["HERMES_KANBAN_TASK"] == tid
        assert env["HERMES_KANBAN_BOARD"] == slug
        assert env["HERMES_KANBAN_DB"] == str(db_path)
        assert env["HERMES_HOME"] == str(profile_home)
        workers[slug] = (tid, env, db_path)

    (tid_a, env_a, db_a), (tid_b, env_b, db_b) = workers["proj-a"], workers["proj-b"]

    # Worker A may not give a verdict on the other board's card, and nothing moves.
    _become(env_a, monkeypatch)
    crossed = _dispatch({"findings": [], "board": "proj-b"})
    assert crossed.get("ok") is not True, crossed
    assert _state(db_a, tid_a)["status"] == "running"
    assert _state(db_b, tid_b)["status"] == "running"

    # On its own board, with no board argument, the verdict approves the card.
    own = _dispatch({"findings": []})
    assert own.get("ok") is True, own
    assert own["outcome"] == "passed"
    approved_a = _state(db_a, tid_a)
    assert (approved_a["status"], approved_a["approvals"], approved_a["deliveries"]) == (
        "done", 1, 0,
    )
    untouched_b = _state(db_b, tid_b)
    assert (untouched_b["status"], untouched_b["approvals"]) == ("running", 0)

    # Worker B, on the second board under the root, names its board and approves.
    _become(env_b, monkeypatch)
    second = _dispatch({"findings": [], "board": "proj-b"})
    assert second.get("ok") is True, second
    assert second["outcome"] == "passed"
    approved_b = _state(db_b, tid_b)
    assert (approved_b["status"], approved_b["approvals"], approved_b["deliveries"]) == (
        "done", 1, 0,
    )
    assert _state(db_a, tid_a)["approvals"] == 1
    assert _release_entries(root) == 0
    # Both verdicts went to the boards under the root, never to a store in the profile home.
    assert not (profile_home / "kanban.db").exists()
    assert not (profile_home / "kanban").exists()
