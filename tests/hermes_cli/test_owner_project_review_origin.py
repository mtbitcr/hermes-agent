"""Keep legacy owner reads, mixed-version writers and worker reads safe with the brief's source references."""

import asyncio
import contextlib
import hashlib
import json
import sqlite3
import sys

import pytest
from aiohttp.test_utils import TestClient, TestServer

from hermes_cli import kanban_db, owner_project_review as review, owner_workspace, projects_db
from tests.gateway.test_api_server_owner_review_advice import accepted_advice, app, stable  # noqa: F401
from tests.hermes_cli.test_owner_project_review_record import root  # noqa: F401
from tests.hermes_cli.test_owner_project_review_suggestions import (  # noqa: F401
    NOW, _commit, _complete, _plan, _request, _result, world,
)
from tests.hermes_cli.test_owner_workspace import ctx  # noqa: F401

HINT = "Ask Raphael in the Project chat to postpone Book the room."
SOURCE_FIELDS = ("source_task_id", "source_run_id", "source_digest")


@pytest.mark.asyncio
async def test_legacy_owner_reads_and_parallel_new_writers_preserve_the_brief(accepted_advice):
    path = accepted_advice["root"] / "projects.db"
    old_fields = ("project_id", "review_date", "covered_from", "covered_to", "headline", "what_changed", "next_step",
                  "needs_you", "ready_work", "cut_note", "quiet", "classification", "written_at", "revision")
    with contextlib.closing(sqlite3.connect(path)) as registry:
        for field in SOURCE_FIELDS:
            registry.execute(f"ALTER TABLE project_review_briefs DROP COLUMN {field}")
        registry.commit()
        old_row = registry.execute(f"SELECT {','.join(old_fields)} FROM project_review_briefs").fetchone()
        before = list(registry.iterdump())
    assert old_row[old_fields.index("next_step")] == HINT
    before_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    endpoint = f"/v1/owner-workspace/projects/{accepted_advice['project'].slug}/snapshot?capabilities=project_summary"
    async with TestClient(TestServer(app())) as client:
        response = await client.get(endpoint)
        assert response.status == 200
        original = (await response.json())["data"]
        assert original["raphael_summary"]["next_action"] == HINT
        with contextlib.closing(sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)) as registry:
            assert list(registry.iterdump()) == before
        assert hashlib.sha256(path.read_bytes()).hexdigest() == before_hash

        # Separate processes have separate initialization caches and share the first-open file lock.
        script = ("from pathlib import Path;import sys;from hermes_cli import projects_db;"
                  "connection=projects_db.connect(Path(sys.argv[1]));connection.close()")
        writers = [await asyncio.create_subprocess_exec(sys.executable, "-c", script, str(path),
                   stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE) for _ in range(2)]
        output = await asyncio.gather(*(writer.communicate() for writer in writers))
        assert [writer.returncode for writer in writers] == [0, 0], output
        with contextlib.closing(sqlite3.connect(path)) as registry:
            columns = list(registry.execute("PRAGMA table_info(project_review_briefs)"))
            assert [(row[1], row[3], row[4]) for row in columns if row[1] in SOURCE_FIELDS] == [
                ("source_task_id", 0, None), ("source_run_id", 0, None), ("source_digest", 0, None),
            ]
            assert registry.execute(f"SELECT {','.join(old_fields)} FROM project_review_briefs").fetchone() == old_row
            assert registry.execute("SELECT source_task_id,source_run_id,source_digest FROM project_review_briefs").fetchone() == (None, None, None)
            migrated = list(registry.iterdump())

        writer = await asyncio.create_subprocess_exec(sys.executable, "-c", script, str(path),
                 stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        output = await writer.communicate()
        assert writer.returncode == 0, output
        with contextlib.closing(sqlite3.connect(path)) as registry:
            assert list(registry.iterdump()) == migrated
            # An old writer names its fourteen columns and does not create a partial source reference.
            registry.execute(f"INSERT OR REPLACE INTO project_review_briefs ({','.join(old_fields)}) "
                             f"VALUES ({','.join('?' for _ in old_fields)})", old_row)
            registry.commit()
            assert registry.execute("SELECT source_task_id,source_run_id,source_digest FROM project_review_briefs").fetchone() == (None, None, None)
            registry.row_factory = sqlite3.Row
            assert projects_db.get_project_review_source(registry, accepted_advice["project"].id) is None
        response = await client.get(endpoint)
        assert response.status == 200
        assert stable((await response.json())["data"]) == stable(original)


def test_a_planner_worker_reads_the_pinned_source_from_the_root_registry_and_the_project_board(accepted_advice, monkeypatch):
    """The worker's own pins name the review board and a profile home; the readers still use the root's stores."""
    from hermes_cli.owner_project_review_origin import accepted_review_source, confirmed_plan_trigger
    from hermes_cli.owner_project_review_view import project_review_advice_view

    data = accepted_advice
    root, project = data["root"], data["project"]
    review_board = root / "kanban" / "boards" / "project-summaries" / "kanban.db"
    # The independent count: plain SQLite reads of the root files, with no Hermes path helper.
    with contextlib.closing(sqlite3.connect(f"{(root / 'projects.db').as_uri()}?mode=ro", uri=True)) as registry:
        pinned = registry.execute("SELECT source_task_id,source_run_id,source_digest FROM project_review_briefs "
                                  "WHERE project_id=? AND source_task_id IS NOT NULL", (project.id,)).fetchall()
    with contextlib.closing(sqlite3.connect(f"{review_board.as_uri()}?mode=ro", uri=True)) as board:
        body = board.execute("SELECT body FROM tasks WHERE id=?", (data["review_task"],)).fetchone()[0]
        runs = board.execute("SELECT id,metadata FROM task_runs WHERE task_id=? AND outcome='completed'",
                             (data["review_task"],)).fetchall()
    assert len(pinned) == 1 and len(runs) == 1
    [(task_id, run_id, digest)] = pinned
    stored = json.loads(runs[0][1])["project_review"]
    assert (task_id, run_id) == (data["review_task"], runs[0][0])
    assert digest == hashlib.sha256(json.dumps({"body": body, "project_review": stored}, sort_keys=True,
                                               ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()

    profile_home = root / "profiles" / "raphael-planner"
    profile_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    monkeypatch.setenv("HERMES_PROFILE", "raphael-planner")
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "project-summaries")
    monkeypatch.setenv("HERMES_KANBAN_DB", str(review_board))
    monkeypatch.setenv("HERMES_KANBAN_TASK", data["review_task"])
    assert projects_db.projects_db_path() == profile_home / "projects.db"
    assert kanban_db.kanban_db_path(board=project.board_slug) == review_board

    accepted = accepted_review_source(project.id)
    assert accepted is not None
    assert (accepted[0].id, accepted[1].id, accepted[2], accepted[3]) == (
        project.id, task_id, stored["plan_proposal"], "2026-11-10")
    advice = project_review_advice_view(project.id)
    assert advice == {"review_date": "2026-11-10", "proposal": stored["plan_proposal"]}
    assert len(advice["proposal"]["changes"]) == len(stored["plan_proposal"]["changes"]) == 1
    # The approved payload's digest, built the way the owner commit builds it, proves the origin.
    proposal = data["proposal"]
    changes, risk = owner_workspace._normalize_project_changes(proposal["changes"])
    expected = {"project_id": project.id, "trigger": "owner_request",
                **{name: proposal[name] for name in ("request_title", "summary", "specification",
                                                     "current_milestone", "owner_visible_result")},
                "later_milestones": owner_workspace._normalize_later_milestones(proposal["later_milestones"]),
                "risk_level": risk, "changes": changes}
    assert confirmed_plan_trigger(project.id, owner_workspace._digest(expected)) == "scheduled_review"
    assert confirmed_plan_trigger(project.id, owner_workspace._digest({**expected, "summary": "Other."})) == "owner_request"
    # Nothing was read from, or created in, the profile home.
    assert not (profile_home / "projects.db").exists()


def _read(path):
    return contextlib.closing(sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True))


def _board_file(root, slug):
    return root / "kanban.db" if slug == "default" else root / "kanban" / "boards" / slug / "kanban.db"


def _counts(world, digest):
    """The independent count: plain read-only SQLite of the root files, with no Hermes path helper."""
    root, accepted = world["root"], world["accepted"]
    with _read(root / "projects.db") as registry:
        denials = registry.execute("SELECT COUNT(*) FROM owner_workspace_receipts WHERE status='denied' "
                                   "AND operation='owner_project_plan_commit' AND request_digest=?", (digest,)).fetchone()[0]
    with _read(_board_file(root, "project-summaries")) as board:
        [(metadata,)] = board.execute("SELECT metadata FROM task_runs WHERE task_id=? AND outcome='completed'",
                                      (world["review_task"],)).fetchall()
    with _read(_board_file(root, world["project"].board_slug)) as board:
        decision = board.execute("SELECT recommendation_decision FROM tasks WHERE id=?", (accepted,)).fetchone()[0]
        links = board.execute("SELECT COUNT(*) FROM recommendation_made_from WHERE recommendation_id=?",
                              (accepted,)).fetchone()[0]
    return {"denials": denials, "stored": json.loads(metadata)["project_review"], "decision": decision, "links": links}


@contextlib.contextmanager
def _worker(world, pinned):
    """The dispatcher's worker environment: its own board and task pinned, and a profile home under the root."""
    root = world["root"]
    slug, task = (("project-summaries", world["review_task"]) if pinned == "project-summaries"
                  else (world["project"].board_slug, world["booked"]))
    profile_home = root / "profiles" / "raphael-planner"
    profile_home.mkdir(parents=True, exist_ok=True)
    with pytest.MonkeyPatch.context() as env:
        env.setenv("HERMES_HOME", str(profile_home))
        env.setenv("HERMES_PROFILE", "raphael-planner")
        env.setenv("HERMES_KANBAN_BOARD", slug)
        env.setenv("HERMES_KANBAN_DB", str(_board_file(root, slug)))
        env.setenv("HERMES_KANBAN_TASK", task)
        assert projects_db.projects_db_path() == profile_home / "projects.db"
        assert {kanban_db.kanban_db_path(board=name) for name in ("project-summaries", world["project"].board_slug)} == {
            _board_file(root, slug)}
        yield
    assert not (profile_home / "projects.db").exists()


@pytest.mark.parametrize("pinned", ["project-summaries", "project"])
def test_t8_a_planner_worker_checks_the_pairs_against_the_root_registry_and_every_board(world, ctx, pinned):
    """The worker's pins name one board and a profile home; the pair check and the commit links still read the root
    registry, the review board and the Project board."""
    from hermes_cli.owner_project_review_origin import confirmed_plan_links
    from hermes_cli.owner_project_review_suggestions import check_plan_suggestions
    from hermes_cli.owner_project_review_view import project_review_advice_view

    project, accepted, plan = world["project"], world["accepted"], _plan()
    _complete(world, _result(world, plan=plan, pairs=[{"change": 0, "suggestion": accepted}]))
    review.review_tick(NOW + 1)
    assert project_review_advice_view(project.id) == {"review_date": "2026-11-10", "proposal": plan}
    assert _commit(ctx, world, plan, key="denied-room-plan", choice="deny")["error"] == "confirmation_denied"
    digest = _request(project.id, plan)
    counts = _counts(world, digest)
    assert (counts["denials"], counts["decision"], counts["links"]) == (1, "accepted", 0)
    pairs = [(pair["change"], pair["suggestion"]) for pair in counts["stored"]["plan_suggestions"]]
    assert pairs == [(0, accepted)]
    with _read(_board_file(world["root"], "project-summaries")) as board:
        sources, _cut = review._frozen_sources(
            board.execute("SELECT body FROM tasks WHERE id=?", (world["review_task"],)).fetchone()[0])

    with _worker(world, pinned):
        assert confirmed_plan_links(project.id, digest) == pairs
        assert confirmed_plan_links(project.id, _request(project.id, {**plan, "summary": "Other."})) == []
        # The owner denied this exact plan: it is not proposed again, to the owner or to a worker.
        assert project_review_advice_view(project.id) is None
        with review._review_evidence_board(project.id) as (board, registry, live), \
                pytest.raises(review.ReviewRefused) as refused:
            check_plan_suggestions(counts["stored"], sources, board, registry, live)
        assert refused.value.code == "review_plan_suggestion_denied"

    # A card made from the suggestion since the review is on the Project board only.
    with kanban_db.connect_closing(_board_file(world["root"], project.board_slug), board=project.board_slug) as board:
        kanban_db.create_task(board, title="Hold the meeting room", project_id=project.id,
                              made_from_recommendation=accepted)
    assert _counts(world, digest)["links"] == 1
    with _worker(world, pinned):
        with review._review_evidence_board(project.id) as (board, registry, live), \
                pytest.raises(review.ReviewRefused) as refused:
            check_plan_suggestions(counts["stored"], sources, board, registry, live)
        assert refused.value.code == "review_plan_suggestion_linked"
