"""Card 10: the delivery step reads the pull request of each armed or unknown arm, and when GitHub shows it merged
into main adds the merge once to the waiting release decision, with the base, head and tree review approved and
the card's recorded tier, title and board. Real boards, git and release store; GitHub is the real transport with
only its exchange and App token replaced by a table."""

from __future__ import annotations

import json
import sqlite3
from urllib.parse import urlsplit

import pytest

from hermes_cli import release_ledger
from tests.hermes_cli.test_kanban_delivery import (  # noqa: F401  (board is a fixture)
    IMPLEMENTER, _become, _git, _spawned_worker_env, _write_config, board,
)
from tests.hermes_cli.test_kanban_delivery_auto_merge import (  # noqa: F401  (armed is a fixture)
    _done, _lenses, _published_as, _reviews, armed,
)
from tests.hermes_cli.test_kanban_delivery_github import REPO
from tests.hermes_cli.test_kanban_delivery_publish_step import _approved, _events, _tick
from tests.hermes_cli.test_kanban_delivery_review_cards import (  # noqa: F401  (world is a fixture)
    _published, _raw, _ready, _state, world,
)


@pytest.fixture
def merged(armed, monkeypatch):
    """The auto-merge world, where GitHub reads each pull request in gh["merges"] closed and merged into main
    as the merge commit given there."""
    kb, root, repo, gh = armed
    from hermes_cli import kanban_delivery_github as transport

    exchange, gh["merges"] = transport._exchange, {}

    def merging(method, target, authorization, payload=None):
        answer = exchange(method, target, authorization, payload)
        parts = urlsplit(target).path.split("/")
        if method == "GET" and len(parts) == 6 and parts[4] == "pulls" and int(parts[5]) in gh["merges"]:
            return 200, dict(answer[1], state="closed", merged=True, base={"ref": "main"},
                             merge_commit_sha=gh["merges"][int(parts[5])])
        return answer

    monkeypatch.setattr(transport, "_exchange", merging)
    return armed


def _merge(repo, head):
    """GitHub's merge commit of H into main, made after another change landed on main, so neither its first
    parent nor its tree is what review approved."""
    (repo / f"{head[:12]}.txt").write_text("a later change\n", encoding="utf-8")
    _git(repo, "add", f"{head[:12]}.txt")
    _git(repo, "commit", "-m", "a later change on main")
    _git(repo, "merge", "--no-ff", "-m", "Merge pull request", head)
    return _git(repo, "rev-parse", "main")


def _merged_head(world, tier=2):
    """H of a card of ``tier``, published as pull request 41 on the bound base, armed once its review cards
    approve it, then merged by GitHub. Returns the card, H, the bound base and the merge commit."""
    kb, root, repo, gh = world
    tid, head = _ready(world, tier=tier)
    base = _git(repo, "rev-parse", "main")
    _tick(kb)
    _done(kb.kanban_db_path())
    _reviews(gh, *_lenses(kb, head))
    _tick(kb)
    assert _state(kb, head) == "auto_merge_armed"
    gh["merges"][41] = merge = _merge(repo, head)
    return tid, head, base, merge


def _store(root, statement="SELECT m.*, b.state FROM release_members AS m JOIN release_batches AS b "
                           "ON b.id = m.batch_id ORDER BY m.rowid"):
    """The release store read straight from its one file at the root: the independent count."""
    path = root / "kanban" / "release_ledger.db"
    if not path.is_file():
        return []
    raw = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
    raw.row_factory = sqlite3.Row
    try:
        return [dict(row) for row in raw.execute(statement)]
    finally:
        raw.close()


def _dispatch(kb, db, board_name):
    """One dispatcher pass on the board whose file is ``db``."""
    ticked = kb.connect(db_path=db)
    try:
        kb.dispatch_once(ticked, board=board_name, spawn_fn=lambda *a, **k: 1)
    finally:
        ticked.close()


@pytest.mark.parametrize("tier, stored", [(1, (1, 1, None)), (None, (2, 0, "tier not recorded"))])
def test_merged_build_joins_the_waiting_decision_with_its_card_tier(merged, tier, stored):
    kb, root, repo, gh = merged
    tid, head, base, merge = _merged_head(merged, tier)
    tree = _git(repo, "rev-parse", f"{head}^{{tree}}")
    assert (_git(repo, "rev-parse", f"{merge}^1"), _git(repo, "rev-parse", f"{merge}^{{tree}}")) != (base, tree)

    _tick(kb)

    assert [(m["merge_commit"], m["pr_url"], m["reviewed_base"], m["reviewed_head"], m["reviewed_tree"], m["tier"],
             m["tier_recorded"], m["card_id"], m["title"], m["board"], m["state"]) for m in _store(root)] == [
        (merge, f"https://github.com/{REPO}/pull/41", base, head, tree, *stored[:2], tid, "build feature",
         "default", "open")]
    conn = release_ledger.connect()
    try:
        assert [m["tier_note"] for batch in release_ledger.list_batches(conn) for m in batch["members"]] == [stored[2]]
    finally:
        conn.close()
    assert _state(kb, head) == "merged"


def test_the_same_merge_seen_twice_is_one_member(merged):
    """The row's merged state lost after its merge was recorded: the next pass sees the same merge again and the
    release record keeps one member. A merged row is final: it is not read again."""
    kb, root, repo, gh = merged
    tid, head, base, merge = _merged_head(merged)
    _tick(kb)
    _raw(kb.kanban_db_path(), "UPDATE kanban_deliveries SET pull_request_state = 'auto_merge_armed' "
         "WHERE source_head = ?", head)

    _tick(kb)
    calls = len(gh["calls"])
    _tick(kb)

    assert [m["merge_commit"] for m in _store(root)] == [merge]
    assert _store(root, "SELECT version FROM release_batches") == [{"version": 1}]
    assert [(p["merge_commit"], p["recorded"]) for kind, p in _events(kb, tid) if kind == "delivery_merged"] == [
        (merge, True), (merge, False)]
    assert (_state(kb, head), len(gh["calls"])) == ("merged", calls)


def test_a_merge_seen_while_a_release_runs_joins_the_next_decision(merged):
    kb, root, repo, gh = merged
    earlier = "1" * 40
    conn = release_ledger.connect()
    try:
        batch = release_ledger.record_merge(
            conn, merge_commit=earlier, pr_url=f"https://github.com/{REPO}/pull/40", reviewed_base="2" * 40,
            reviewed_head="3" * 40, reviewed_tree="4" * 40, tier=0, card_id="t_earlier", on_main=True)["batch"]
        release_ledger.decide_release(conn, batch["batch_id"], decision="accepted", shown_digest=batch["digest"],
                                      expected_version=batch["version"], decision_ref="decisions-page:release")
        release_ledger.begin_release(conn, batch["batch_id"], prev="5" * 40, new=earlier)
    finally:
        conn.close()
    tid, head, base, merge = _merged_head(merged)

    _tick(kb)

    assert [(m["batch_id"], m["state"], m["merge_commit"]) for m in _store(root)] == [
        (1, "releasing", earlier), (2, "open", merge)]
    assert _state(kb, head) == "merged"


def test_record_failure_never_breaks_delivery_and_is_retried(merged):
    """The root store cannot be opened (its path is a directory): the merge is not recorded and its row stays
    armed, while the same passes still arm another head. The next pass with the store back records it once."""
    kb, root, repo, gh = merged
    db, store = kb.kanban_db_path(), root / "kanban" / "release_ledger.db"
    store.mkdir(parents=True)
    tid, head, base, merge = _merged_head(merged)
    second, later = _approved(kb, repo, "second")
    _published_as(db, gh, second, later, 42)
    _tick(kb)
    _done(db)
    _reviews(gh, *_lenses(kb, later), number=42)

    _tick(kb)

    assert (_state(kb, head), _state(kb, later), store.is_dir()) == ("auto_merge_armed", "auto_merge_armed", True)
    assert [kind for kind, _ in _events(kb, tid) if kind == "delivery_merged"] == []
    store.rmdir()

    _tick(kb)

    assert [(m["merge_commit"], m["reviewed_head"]) for m in _store(root)] == [(merge, head)]
    assert (_state(kb, head), _state(kb, later)) == ("merged", "auto_merge_armed")


def test_intake_reaches_the_root_store_under_a_worker_env(merged, monkeypatch):
    """The dispatcher's worker environment for proj-a (HERMES_KANBAN_DB, HERMES_KANBAN_BOARD and
    HERMES_KANBAN_TASK pinned to proj-a, HERMES_HOME a profile home under the root) ticks the board other and
    the root board: each merge lands in the one release store at the root, counted straight from its file."""
    kb, root, repo, gh = merged
    kb.create_board("proj-a")
    kb.create_board("other")
    own_db = root / "kanban" / "boards" / "proj-a" / "kanban.db"
    other_db = root / "kanban" / "boards" / "other" / "kanban.db"
    root_db = root / "kanban.db"
    profile_home = root / "profiles" / IMPLEMENTER
    profile_home.mkdir(parents=True)
    _write_config(profile_home, enabled=True)
    tid, head = _approved(kb, repo, "beta", board="other")
    _published(kb, repo, gh, tid, head, board="other")
    _raw(other_db, "UPDATE tasks SET risk_tier = 1 WHERE id = ?", tid)
    home, home_head = _approved(kb, repo, "gamma")
    _published_as(root_db, gh, home, home_head, 44)
    _raw(root_db, "INSERT INTO task_events (task_id, kind, payload, created_at) VALUES (?, 'delivery_bound', ?, 0)",
         home, json.dumps({"source_task_id": home, "head": home_head, "base_commit": _git(repo, "rev-parse", "main")}))
    boards = ((other_db, "other"), (root_db, kb.DEFAULT_BOARD))
    for db, board_name in boards:
        _dispatch(kb, db, board_name)
        _done(db)
    _reviews(gh, *_lenses(kb, head, db=other_db))
    _reviews(gh, *_lenses(kb, home_head, db=root_db), number=44)
    for db, board_name in boards:
        _dispatch(kb, db, board_name)
    gh["merges"].update({41: _merge(repo, head), 44: _merge(repo, home_head)})
    conn = kb.connect(board="proj-a")
    try:
        claimed = kb.claim_task(conn, kb.create_task(conn, title="plain work", assignee=IMPLEMENTER))
        workspace = kb.resolve_workspace(claimed, board="proj-a")
        kb.set_workspace_path(conn, claimed.id, str(workspace))
        env = _spawned_worker_env(kb, claimed, workspace, "proj-a", monkeypatch)
    finally:
        conn.close()
    assert (env["HERMES_KANBAN_TASK"], env["HERMES_KANBAN_DB"], env["HERMES_KANBAN_BOARD"], env["HERMES_HOME"]) == (
        claimed.id, str(own_db), "proj-a", str(profile_home))

    _become(env, monkeypatch)
    for db, board_name in boards:
        _dispatch(kb, db, board_name)

    assert sorted((m["card_id"], m["board"], m["tier"], m["merge_commit"]) for m in _store(root)) == sorted([
        (tid, "other", 1, gh["merges"][41]), (home, kb.DEFAULT_BOARD, 2, gh["merges"][44])])
    assert [path for path in root.rglob("*") if "release" in path.name] == [root / "kanban" / "release_ledger.db"]
    assert (_state(kb, head, other_db), _state(kb, home_head, root_db)) == ("merged", "merged")
