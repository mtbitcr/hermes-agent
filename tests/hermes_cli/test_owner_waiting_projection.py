"""The owner pages tell the truth about waiting and stopped work.

Every new answer is served only to a reader that names the
``owner_waiting_v1`` response capability; a reader that does not name it gets
exactly today's shape and selection. Each test builds a real board in its own
temporary home through the real kernel calls and reads it back through the
same projections the owner Workspace is served from.
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

from hermes_cli import kanban_db, owner_workspace as ow

_WORKER = "raphael-worker"
_REVIEWER = "raphael-reviewer"
_SECRET = "sk-" + "a" * 24
_PATH = "/srv/placeholder/config.yaml"
_FILE_LINK = "file:///srv/placeholder/index.html"
_INTERNAL_ID = "task_1234abcd"
_KERNEL_TASK_ID = "t_0000abcd"
_PROJECT_ID = "p_0000abcd"
_HOME_PATH = "~/notes.txt"
_COMMIT_ID = "0000abcd" * 5
_DECISION_FALLBACK = "Raphael needs your answer before this work can continue."
_RECEIPT_FALLBACK = "Work stopped and needs attention."
_REWORK_SUMMARY = "The review asked for changes; the work went back for rework."


class _Clock:
    """The kernel's own clock, pinned so recorded times are exact."""

    def __init__(self, now: int) -> None:
        self.now = now

    def time(self) -> float:
        return float(self.now)

    def __getattr__(self, name):
        return getattr(time, name)


@pytest.fixture
def owner() -> ow.OwnerContext:
    return ow.resolve_owner_context()


@pytest.fixture
def clock(monkeypatch) -> _Clock:
    fake = _Clock(int(time.time()) - 3_600)
    monkeypatch.setattr(kanban_db, "time", fake)
    return fake


def _iso(value: int) -> str:
    return datetime.fromtimestamp(value, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _project(owner: ow.OwnerContext, name: str) -> dict:
    """Commit one owner Project whose board may dispatch."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(ow, "_confirm", lambda *_args, **_kwargs: {"approved": True})
        project = ow.bootstrap(owner, idempotency_key=f"setup-{name}", name=name)
    assert project["ok"] is True
    kanban_db.write_board_dispatch_state(project["board"], dispatch_enabled=True)
    project["slug"] = next(
        item["slug"] for item in ow.list_committed_projects(owner)
        if str(item["project_id"]) == project["project_id"]
    )
    return project


def _board(project: dict):
    return contextlib.closing(kanban_db.connect(board=project["board"]))


def _card(conn, project: dict, title: str, **kwargs) -> str:
    return kanban_db.create_task(
        conn, title=title, assignee=_WORKER,
        project_id=project["project_id"], **kwargs,
    )


def _stopped(conn, project: dict, title: str, *, kind: str, reason: str) -> str:
    """A card a worker ran and then stopped with a typed reason."""
    task_id = _card(conn, project, title)
    assert kanban_db.claim_task(conn, task_id) is not None
    assert kanban_db.block_task(conn, task_id, reason=reason, kind=kind) is True
    return task_id


def _gave_up(conn, project: dict, title: str) -> str:
    """A card the repeated-failure breaker stopped."""
    task_id = _card(conn, project, title)
    assert kanban_db.claim_task(conn, task_id) is not None
    assert kanban_db._record_task_failure(
        conn, task_id, "worker crashed repeatedly",
        outcome="crashed", failure_limit=1,
        release_claim=False, end_run=False,
    ) is True
    assert kanban_db.get_task(conn, task_id).status == "blocked"
    return task_id


def _owner_move(conn, task_id: str, *, to_status: str) -> None:
    """The owner's own compare-and-swap move, as the owner kernel records it."""
    task = kanban_db.get_task(conn, task_id)
    moved = kanban_db.cas_transition_task(
        conn, task_id,
        expected_status=task.status,
        expected_revision=kanban_db.task_event_revision(conn, task_id),
        to_status=to_status,
        event_kind="owner_move",
        event_payload={"actor": "owner", "to_status": to_status},
    )
    assert moved["moved"] is True, moved


def _park_for_review(conn, task_id: str) -> None:
    task = kanban_db.get_task(conn, task_id)
    assert kanban_db.request_review(
        conn, task_id, summary="candidate ready", reviewer=_REVIEWER,
        expected_run_id=task.current_run_id,
    ) is True


def _returned(conn, task_id: str) -> None:
    """The reviewer claims the parked card and asks for changes."""
    review = kanban_db.claim_review_task(conn, task_id)
    assert review is not None
    ok, _who = kanban_db.request_changes(
        conn, task_id, reason="Tighten the placeholder section",
        expected_run_id=review.current_run_id,
    )
    assert ok is True


def _vouched_rework(conn, task_id: str) -> list[str]:
    """The rework cards the kernel vouched for, read from the raw events."""
    ids = []
    for row in conn.execute(
        "SELECT payload FROM task_events "
        "WHERE task_id = ? AND kind = 'review_followup_recorded' ORDER BY id",
        (task_id,),
    ):
        followup = json.loads(row["payload"]).get("followup_task_id")
        if followup not in ids:
            ids.append(followup)
    return ids


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


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(
        ["git", "init", "-b", "main", str(repo)],
        check=True, capture_output=True, text=True,
    )
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "init")
    return repo


def _commit_version(workspace: Path, version: int) -> str:
    path = workspace / "src" / "guide" / "draft.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"placeholder draft {version}\n", encoding="utf-8")
    _git(workspace, "add", "src/guide/draft.md")
    _git(workspace, "commit", "-m", f"draft {version}")
    return _git(workspace, "rev-parse", "HEAD")


def _decisions_for(owner, project: dict, **kwargs) -> list[dict]:
    return [
        item for item in ow.list_owner_decisions(owner, **kwargs)["data"]
        if item["project_slug"] == project["slug"]
    ]


def _tasks_by_title(snapshot: dict) -> dict[str, dict]:
    return {
        task["title"]: task
        for column in snapshot["columns"]
        for task in column["tasks"]
    }


def test_active_work_lists_running_then_review_then_ready_and_planned_is_not_complete(
    owner, clock,
):
    """P1: running, review, ready in that order, most recently started first."""
    project = _project(owner, "Workshop Pilot")
    with _board(project) as conn:
        gate = _card(conn, project, "Book the placeholder venue")
        _card(conn, project, "Plan the placeholder follow-up", parents=[gate])
        scheduled = _card(conn, project, "Send the placeholder reminder")
        assert kanban_db.schedule_task(
            conn, scheduled, reason="Wait for the placeholder date",
        ) is True
        checked = _card(conn, project, "Check the placeholder draft")
        clock.now += 10
        assert kanban_db.claim_task(conn, checked) is not None
        _park_for_review(conn, checked)
        older = _card(conn, project, "Collect the placeholder notes")
        clock.now += 10
        assert kanban_db.claim_task(conn, older) is not None
        newer = _card(conn, project, "Write the placeholder summary")
        clock.now += 10
        assert kanban_db.claim_task(conn, newer) is not None
        statuses = {
            row["title"]: row["status"]
            for row in conn.execute(
                "SELECT title, status FROM tasks WHERE project_id = ? "
                "AND task_kind = 'work'",
                (project["project_id"],),
            )
        }
    assert sorted(statuses.values()) == [
        "ready", "review", "running", "running", "scheduled", "todo",
    ]

    steward = ow.read_project_snapshot(
        owner, project["slug"], owner_waiting=True,
    )["steward"]

    assert steward["active_work"] == [
        {"title": "Write the placeholder summary", "state": "In progress"},
        {"title": "Collect the placeholder notes", "state": "In progress"},
        {"title": "Check the placeholder draft", "state": "Being checked"},
        {"title": "Book the placeholder venue", "state": "Ready"},
    ]
    assert steward["truncated"]["active_work"] is False
    assert steward["execution"]["state"] == "working"

    # A board holding only planned and scheduled work is not complete.
    quiet = _project(owner, "Quiet Pilot")
    with _board(quiet) as conn:
        later = _card(conn, quiet, "Order the placeholder supplies")
        assert kanban_db.schedule_task(conn, later, reason="Next month") is True
        _card(conn, quiet, "Unpack the placeholder supplies", parents=[later])
    quiet_steward = ow.project_steward_snapshot(
        project_id=quiet["project_id"], owner_waiting=True,
    )
    assert quiet_steward["active_work"] == []
    assert quiet_steward["execution"]["state"] == "working"
    assert quiet_steward["execution"]["state"] != "complete"


def test_returned_reviews_are_rework_not_progress_or_attention(owner, tmp_path):
    """P2: two returns on two versions, then an approval."""
    project = _project(owner, "Guide Pilot")
    repo = _repo(tmp_path)
    with _board(project) as conn:
        guide = _card(
            conn, project, "Write the placeholder guide",
            workspace_kind="worktree", workspace_path=str(repo),
            branch_name="feature/placeholder-guide", owned_paths=["src/guide"],
        )
        claimed = kanban_db.claim_task(conn, guide)
        assert claimed is not None
        workspace, branch = kanban_db._resolve_worktree_workspace(claimed)
        kanban_db.set_workspace_path(conn, guide, workspace)
        kanban_db.set_branch_name(conn, guide, branch)
        kanban_db.record_worktree_base(conn, guide, workspace)
        heads = []
        for version in (1, 2):
            heads.append(_commit_version(Path(workspace), version))
            _park_for_review(conn, guide)
            _returned(conn, guide)
            assert kanban_db.claim_task(conn, guide) is not None
        assert heads[0] != heads[1]
        _commit_version(Path(workspace), 3)
        _park_for_review(conn, guide)
        rework = _vouched_rework(conn, guide)
        assert len(rework) == 2
        assert {kanban_db.get_task(conn, item).status for item in rework} == {"triage"}

        # A card someone merely TITLED like rework is ordinary work.
        lookalike = _card(conn, project, "Rework: Tidy the placeholder index")
        assert kanban_db.claim_task(conn, lookalike) is not None
        assert kanban_db.complete_task(conn, lookalike, summary="tidied") is True

    open_snapshot = ow.read_project_snapshot(
        owner, project["slug"], owner_waiting=True,
    )
    steward = open_snapshot["steward"]
    # Open rework cards raise no attention and read as no problem.
    assert steward["needs_attention"] == []
    assert steward["counts"]["needs_attention"] == 0
    assert steward["execution"]["state"] == "working"
    # A returned review's receipt says rework, with its outcome unchanged.
    returned = [
        run["receipt"] for run in open_snapshot["runs"]
        if run["receipt"]["summary"] == _REWORK_SUMMARY
    ]
    assert len(returned) == 2
    assert {receipt["outcome"] for receipt in returned} == {"attention"}
    assert all(
        run["receipt"]["summary"] != _RECEIPT_FALLBACK
        for run in open_snapshot["runs"]
    )

    with _board(project) as conn:
        review = kanban_db.claim_review_task(conn, guide)
        assert review is not None
        verdict = kanban_db.submit_review_findings(
            conn, guide, findings=[], candidate_digest="placeholder-clean",
            expected_run_id=review.current_run_id,
        )
        assert verdict["outcome"] == "passed", verdict
        assert {kanban_db.get_task(conn, item).status for item in rework} == {"done"}

    steward = ow.project_steward_snapshot(
        project_id=project["project_id"], owner_waiting=True,
    )
    assert [item["title"] for item in steward["progress"]] == [
        "Write the placeholder guide", "Rework: Tidy the placeholder index",
    ] or [item["title"] for item in steward["progress"]] == [
        "Rework: Tidy the placeholder index", "Write the placeholder guide",
    ]
    assert steward["counts"]["completed_in_window"] == 2
    assert steward["needs_attention"] == []


def test_a_capability_stop_is_an_owner_decision_with_its_cleaned_reason(owner):
    """P3: a capability stop asks the owner; a breaker stop does not."""
    project = _project(owner, "Venue Pilot")
    reason = "Please connect the placeholder calendar account so I can book the room."
    with _board(project) as conn:
        _stopped(
            conn, project, "Book the placeholder room", kind="capability",
            reason=reason,
        )
        _stopped(
            conn, project, "Deploy the placeholder page", kind="capability",
            reason=f"Need the deploy key {_SECRET} to continue",
        )
        _stopped(
            conn, project, "Read the placeholder settings", kind="capability",
            reason=f"Cannot open {_PATH} for reading",
        )
        _stopped(
            conn, project, "Resume the placeholder import", kind="capability",
            reason=f"Waiting on {_INTERNAL_ID} before continuing",
        )
        _gave_up(conn, project, "Retry the placeholder upload")

    decisions = {
        item["title"]: item
        for item in _decisions_for(owner, project, owner_waiting=True)
    }
    assert set(decisions) == {
        "Book the placeholder room", "Deploy the placeholder page",
        "Read the placeholder settings", "Resume the placeholder import",
    }
    assert {(item["authority"], item["kind"]) for item in decisions.values()} == {
        ("task", "owner_input"),
    }
    assert decisions["Book the placeholder room"]["reason"] == reason
    for title in (
        "Deploy the placeholder page", "Read the placeholder settings",
        "Resume the placeholder import",
    ):
        assert decisions[title]["reason"] == _DECISION_FALLBACK
    assert all(
        item["decision_ref"].startswith("decision_") for item in decisions.values()
    )

    steward = ow.project_steward_snapshot(
        project_id=project["project_id"], owner_waiting=True,
    )
    assert steward["execution"] == {
        "state": "waiting_for_you",
        "summary": "Raphael needs your answer before the plan can continue.",
        "paused": False,
    }
    assert {item["title"] for item in steward["decisions_needed"]} == set(decisions)
    # Only the breaker stop stays under attention.
    assert [item["title"] for item in steward["needs_attention"]] == [
        "Retry the placeholder upload",
    ]
    assert steward["counts"]["needs_attention"] == 1

    # The helper itself: bounded, cleaned, and never showing private detail.
    assert ow.owner_stop_reason(reason) == reason
    long_reason = "Please confirm the placeholder plan. " * 30
    assert ow.owner_stop_reason(long_reason) == long_reason[:500].strip()
    for private in (
        f"Need the deploy key {_SECRET} to continue",
        f"Cannot open {_PATH} for reading",
        f"Waiting on {_INTERNAL_ID} before continuing",
        "",
        None,
    ):
        assert ow.owner_stop_reason(private) == _DECISION_FALLBACK
        assert ow.owner_stop_reason(
            private, fallback=_RECEIPT_FALLBACK,
        ) == _RECEIPT_FALLBACK

    # The ids and error text workers really write, on both kinds of stop,
    # fall back whole on every owner surface; a question without them still
    # comes through.
    leaks = {
        "approval": f"Waiting for the owner to approve {_KERNEL_TASK_ID} before booking",
        "settings": f"Cannot read {_PATH}: permission denied",
        "import": "[Errno 2] No such file or directory: '/srv/placeholder/x'",
        "notes": f"Cannot open {_HOME_PATH} for reading",
        "draft": f"Should I publish the draft at commit {_COMMIT_ID}?",
    }
    question = "Should we book it for the fifth or the nineteenth?"
    # A file: link and this platform's own Project id fall back the same way,
    # and a question about a file still comes through. They get a second
    # Project because the steward lists at most 12 decisions per Project.
    link_leaks = {
        "page": f"Cannot open {_FILE_LINK}: the browser is not available",
        "move": f"Should I move this into {_PROJECT_ID} first?",
    }
    link_question = "Which file: the first or the second?"
    # A path with a doubled slash, or one written without its leading slash,
    # falls back the same way.
    path_leaks = {
        "share": "Cannot open //srv/placeholder/x for reading",
        "doubled": "The upload failed (see /srv//placeholder/x)",
        "relative": "Saved the list to srv/placeholder/list.txt, please check it",
        "file": "Please check config/settings.py",
        "windows": "Cannot read config\\settings.py",
    }
    path_question = "Should we book it for next Monday or next Tuesday?"
    # A file name's extension may be long, start with a digit, or be the whole
    # name; a short relative path reads like a choice.
    name_leaks = {
        "long": "Please check config/application.properties",
        "hidden": "Please check config/.sample",
        "numeric": "Please check config/archive.7z",
        "spaced": "Please check config/my notes.py",
        "choice": "Cannot open A/B: permission denied",
    }
    name_question = "Should the placeholder room be booked for two hours?"
    for project_name, project_leaks, question_title, project_question in (
        ("Worker Text Pilot", leaks, "Choose the placeholder date", question),
        ("Worker Link Pilot", link_leaks, "Choose the placeholder file", link_question),
        ("Worker Path Pilot", path_leaks, "Choose the placeholder day", path_question),
        ("Worker Name Pilot", name_leaks, "Choose the placeholder hours", name_question),
    ):
        worker_text = _project(owner, project_name)
        leaked = []
        with _board(worker_text) as conn:
            for kind in ("capability", "needs_input"):
                for name, leak in project_leaks.items():
                    title = f"Check the placeholder {name} ({kind.replace('_', ' ')})"
                    _stopped(conn, worker_text, title, kind=kind, reason=leak)
                    leaked.append(title)
            _stopped(
                conn, worker_text, question_title, kind="needs_input",
                reason=project_question,
            )
        shown = dict.fromkeys(leaked, _DECISION_FALLBACK) | {
            question_title: project_question,
        }

        assert {
            item["title"]: (item["kind"], item["reason"])
            for item in _decisions_for(owner, worker_text, owner_waiting=True)
        } == {title: ("owner_input", text) for title, text in shown.items()}
        worker_steward = ow.project_steward_snapshot(
            project_id=worker_text["project_id"], owner_waiting=True,
        )
        assert {
            item["title"]: item["reason"] for item in worker_steward["decisions_needed"]
        } == shown
        worker_snapshot = ow.read_project_snapshot(
            owner, worker_text["slug"], run_context=True, owner_waiting=True,
        )
        tasks = _tasks_by_title(worker_snapshot)
        assert {title: tasks[title]["owner_wait"]["reason"] for title in shown} == shown
        stop_reasons: dict[str, list] = {}
        for run in worker_snapshot["runs"]:
            stop_reasons.setdefault(run["task_title"], []).append(
                run["receipt"].get("stop_reason")
            )
        assert stop_reasons == {title: [_RECEIPT_FALLBACK] for title in leaked} | {
            question_title: [project_question],
        }

    for private in (
        *leaks.values(),
        *link_leaks.values(),
        *path_leaks.values(),
        "Open ./settings.py first",
        "The draft is in notes\\placeholder\\drafts",
        "Saved it as 'notes/list.txt' for you",
        "Look in \u043a\u043e\u043d\u0444\u0438\u0433/\u0444\u0430\u0439\u043b for it",
        "See https://example.test/help/owner for the steps",
        "Saved the placeholder list to ~/placeholder/list.txt, please check it",
        "The upload failed (see /srv/placeholder/x)",
        "Should I publish file:///srv/placeholder/draft.html as it is?",
        "Project p_0000abcd0000abcd0000abcd has no repository configured",
        # Any slash may be a path, so none is explained away: dates, versions,
        # choices, and look-alikes of both separators fall back too.
        "Should we book it for 12/05 or 19/05?",
        "Book it for 12/05/2026.", "Python 3.11/3.12 both work", "Use v1.2/v1.3",
        "Plan the A/B test", "Is it yes/no.", "Is it and/or.",
        "Cannot open A∕B: permission denied",
        "Cannot open A／B: permission denied",
        "The draft is in notes＼drafts",
    ):
        assert ow.owner_stop_reason(private) == _DECISION_FALLBACK
        assert ow.owner_stop_reason(
            private, fallback=_RECEIPT_FALLBACK,
        ) == _RECEIPT_FALLBACK
    # Neither fixed sentence, nor an ordinary question, trips a pattern.
    for ordinary in (
        reason, question, link_question, path_question, "Should p_values be reported?",
        "Is it yes or no?", "Book it for the fifth of December.",
        _DECISION_FALLBACK, _RECEIPT_FALLBACK,
    ):
        for fallback in (_DECISION_FALLBACK, _RECEIPT_FALLBACK):
            assert ow.owner_stop_reason(ordinary, fallback=fallback) == ordinary


def test_a_stopped_run_receipt_carries_its_stop_reason(owner):
    """P3: only a blocked run's receipt gains stop_reason."""
    project = _project(owner, "Receipt Pilot")
    reason = "Please connect the placeholder calendar account."
    with _board(project) as conn:
        _stopped(
            conn, project, "Book the placeholder room", kind="capability",
            reason=reason,
        )
        _stopped(
            conn, project, "Ask about the placeholder budget", kind="needs_input",
            reason=f"Is {_SECRET} the right key?",
        )
        finished = _card(conn, project, "Send the placeholder invite")
        assert kanban_db.claim_task(conn, finished) is not None
        assert kanban_db.complete_task(conn, finished, summary="sent") is True
        _gave_up(conn, project, "Retry the placeholder upload")

    snapshot = ow.read_project_snapshot(
        owner, project["slug"], run_context=True, owner_waiting=True,
    )
    receipts: dict[str, list[dict]] = {}
    for run in snapshot["runs"]:
        receipts.setdefault(run["task_title"], []).append(run["receipt"])
    [stopped] = receipts["Book the placeholder room"]
    assert stopped["stop_reason"] == reason
    assert stopped["summary"] == _RECEIPT_FALLBACK
    [private] = receipts["Ask about the placeholder budget"]
    assert private["stop_reason"] == _RECEIPT_FALLBACK
    for title in ("Send the placeholder invite", "Retry the placeholder upload"):
        assert receipts[title]
        assert all("stop_reason" not in receipt for receipt in receipts[title])


def test_owner_wait_marks_only_cards_still_waiting_on_the_owner(owner):
    """P4: owner_wait follows the stop until it is answered or moved."""
    project = _project(owner, "Plan Pilot")
    question = "Which placeholder venue should we book?"
    with _board(project) as conn:
        waiting = _stopped(
            conn, project, "Choose the placeholder venue", kind="needs_input",
            reason=question,
        )
        answered = _stopped(
            conn, project, "Choose the placeholder date", kind="needs_input",
            reason="Which placeholder date?",
        )
        assert kanban_db.unblock_task(conn, answered) is True
        moved = _stopped(
            conn, project, "Choose the placeholder menu", kind="needs_input",
            reason="Which placeholder menu?",
        )
        _owner_move(conn, moved, to_status="todo")
        # Moved back by the owner: blocked again, but by the owner's hand.
        _owner_move(conn, moved, to_status="blocked")
        assert kanban_db.get_task(conn, moved).status == "blocked"
        stopped_at = conn.execute(
            "SELECT created_at FROM task_events WHERE task_id = ? "
            "AND kind = 'blocked' ORDER BY id DESC LIMIT 1",
            (waiting,),
        ).fetchone()["created_at"]
        waits = kanban_db.task_owner_waits(conn, [waiting, answered, moved])

    assert waits == {
        waiting: {"kind": "needs_input", "since": stopped_at, "reason": question},
    }
    tasks = _tasks_by_title(
        ow.read_project_snapshot(owner, project["slug"], owner_waiting=True)
    )
    assert tasks["Choose the placeholder venue"]["owner_wait"] == {
        "since": _iso(stopped_at), "reason": question,
    }
    assert "owner_wait" not in tasks["Choose the placeholder date"]
    assert "owner_wait" not in tasks["Choose the placeholder menu"]


def test_waiting_since_is_the_stop_time_not_the_creation_time(owner, clock):
    """P4: Decisions created_at and steward waiting_since are the stop time."""
    project = _project(owner, "Timing Pilot")
    created = clock.now
    with _board(project) as conn:
        question = _card(conn, project, "Choose the placeholder venue")
        capability = _card(conn, project, "Book the placeholder room")
        moved = _card(conn, project, "Choose the placeholder menu")
        clock.now = created + 900
        assert kanban_db.claim_task(conn, question) is not None
        assert kanban_db.block_task(
            conn, question, reason="Which placeholder venue?", kind="needs_input",
        ) is True
        clock.now = created + 1_200
        assert kanban_db.claim_task(conn, capability) is not None
        assert kanban_db.block_task(
            conn, capability, reason="Please connect the placeholder calendar.",
            kind="capability",
        ) is True
        clock.now = created + 1_500
        assert kanban_db.claim_task(conn, moved) is not None
        assert kanban_db.block_task(
            conn, moved, reason="Which placeholder menu?", kind="needs_input",
        ) is True
        clock.now = created + 1_600
        _owner_move(conn, moved, to_status="todo")
        _owner_move(conn, moved, to_status="blocked")

    decisions = {
        item["title"]: item
        for item in _decisions_for(owner, project, owner_waiting=True)
    }
    assert decisions["Choose the placeholder venue"]["created_at"] == _iso(created + 900)
    assert decisions["Choose the placeholder venue"]["reason"] == "Which placeholder venue?"
    assert decisions["Book the placeholder room"]["created_at"] == _iso(created + 1_200)
    # No current stop event: the creation time and the fixed sentence.
    assert decisions["Choose the placeholder menu"]["created_at"] == _iso(created)
    assert decisions["Choose the placeholder menu"]["reason"] == _DECISION_FALLBACK

    steward = ow.project_steward_snapshot(
        project_id=project["project_id"], owner_waiting=True,
    )
    needed = {item["title"]: item for item in steward["decisions_needed"]}
    assert needed["Choose the placeholder venue"] == {
        "title": "Choose the placeholder venue",
        "state": "Waiting for your answer",
        "reason": "Which placeholder venue?",
        "waiting_since": _iso(created + 900),
    }
    assert needed["Book the placeholder room"]["waiting_since"] == _iso(created + 1_200)
    assert needed["Choose the placeholder menu"]["waiting_since"] == _iso(created)
    assert needed["Choose the placeholder menu"]["reason"] == _DECISION_FALLBACK


def test_without_the_capability_every_output_keeps_todays_shape_and_selection(owner):
    """A reader that does not name owner_waiting_v1 sees today's answers."""
    project = _project(owner, "Legacy Pilot")
    with _board(project) as conn:
        gate = _card(conn, project, "Book the placeholder venue")
        _card(conn, project, "Plan the placeholder follow-up", parents=[gate])
        scheduled = _card(conn, project, "Send the placeholder reminder")
        assert kanban_db.schedule_task(conn, scheduled, reason="Later") is True
        _stopped(
            conn, project, "Book the placeholder room", kind="capability",
            reason="Please connect the placeholder calendar account.",
        )
        _stopped(
            conn, project, "Choose the placeholder menu", kind="needs_input",
            reason="Which placeholder menu?",
        )
        reviewed = _card(conn, project, "Check the placeholder draft")
        assert kanban_db.claim_task(conn, reviewed) is not None
        _park_for_review(conn, reviewed)
        _returned(conn, reviewed)
        [rework] = _vouched_rework(conn, reviewed)
        assert kanban_db.get_task(conn, rework).status == "triage"

    snapshot = ow.read_project_snapshot(owner, project["slug"])
    steward = snapshot["steward"]
    assert steward == ow.project_steward_snapshot(
        project_id=project["project_id"], lookback_days=7,
    ) | {"generated_at": steward["generated_at"]}
    # Today's selection: planned and scheduled count as active work, the
    # capability stop and the open rework card count as attention.
    assert sorted(item["title"] for item in steward["active_work"]) == [
        "Book the placeholder venue", "Check the placeholder draft",
        "Plan the placeholder follow-up", "Send the placeholder reminder",
    ]
    assert {item["title"] for item in steward["needs_attention"]} == {
        "Book the placeholder room", "Rework: Check the placeholder draft",
    }
    assert steward["decisions_needed"] == [
        {"title": "Choose the placeholder menu", "state": "Waiting for your answer"},
    ]
    for task in _tasks_by_title(snapshot).values():
        assert "owner_wait" not in task
    for run in snapshot["runs"]:
        assert set(run) == {"started_at", "finished_at", "receipt"}
        assert "stop_reason" not in run["receipt"]
        assert run["receipt"]["summary"] != _REWORK_SUMMARY
    assert _RECEIPT_FALLBACK in {run["receipt"]["summary"] for run in snapshot["runs"]}

    decisions = _decisions_for(owner, project)
    assert [(item["title"], item["kind"], item["reason"]) for item in decisions] == [
        ("Choose the placeholder menu", "owner_input", _DECISION_FALLBACK),
    ]
    assert set(decisions[0]) == {
        "decision_ref", "authority", "kind", "project_slug", "project_name",
        "title", "reason", "created_at",
    }


def test_a_worker_pinned_to_its_board_reads_every_projects_waits(
    tmp_path, monkeypatch,
):
    """The production proof, read under the environment a worker really gets.

    The dispatcher pins a worker to its own board (``HERMES_KANBAN_DB`` +
    ``HERMES_KANBAN_BOARD``), names its task (``HERMES_KANBAN_TASK``) and points
    ``HERMES_HOME`` at a profile home under the shared root. The steward read
    and the Decisions list must still reach the OTHER Project's board and the
    root register from there.
    """
    root = tmp_path / "hermes_root"
    profile_home = root / "profiles" / "worker"
    profile_home.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(root))
    kanban_db._INITIALIZED_PATHS.clear()
    kanban_db.init_db()
    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    owner = ow.resolve_owner_context()
    first = _project(owner, "First Pilot")
    second = _project(owner, "Second Pilot")
    with _board(first) as conn:
        pinned = _stopped(
            conn, first, "Book the placeholder room", kind="capability",
            reason="Please connect the placeholder calendar.",
        )
    with _board(second) as conn:
        _stopped(
            conn, second, "Choose the placeholder venue", kind="needs_input",
            reason="Which placeholder venue?",
        )
        _stopped(
            conn, second, "Book the placeholder hall", kind="capability",
            reason="Please connect the placeholder booking account.",
        )
        _gave_up(conn, second, "Retry the placeholder upload")

    paths = {
        project["slug"]: kanban_db.board_dir(project["board"]) / "kanban.db"
        for project in (first, second)
    }

    # Exactly what ``_default_spawn`` injects into the worker subprocess.
    board_db = paths[first["slug"]]
    monkeypatch.setenv("HERMES_KANBAN_DB", str(board_db))
    monkeypatch.setenv("HERMES_KANBAN_BOARD", first["board"])
    monkeypatch.setenv("HERMES_KANBAN_TASK", pinned)
    with contextlib.closing(kanban_db.connect()) as conn:
        opened = conn.execute("PRAGMA database_list").fetchone()[2]
    assert os.path.realpath(opened) == os.path.realpath(board_db)

    decisions = ow.list_owner_decisions(owner, owner_waiting=True)["data"]
    stewards = {
        project["slug"]: ow.project_steward_snapshot(
            project_id=project["project_id"], owner_waiting=True,
        )
        for project in (first, second)
    }

    # The independent count, straight from each board's own rows.
    expected: dict[str, int] = {}
    for project in (first, second):
        path = paths[project["slug"]]
        with contextlib.closing(kanban_db.connect(db_path=path)) as conn:
            expected[project["slug"]] = conn.execute(
                "SELECT COUNT(*) AS n FROM tasks WHERE project_id = ? "
                "AND task_kind = 'work' AND status = 'blocked' "
                "AND block_kind IN ('needs_input', 'capability')",
                (project["project_id"],),
            ).fetchone()["n"]
    assert expected == {first["slug"]: 1, second["slug"]: 2}
    for slug, count in expected.items():
        assert sum(
            item["project_slug"] == slug and item["kind"] == "owner_input"
            for item in decisions
        ) == count
        assert len(stewards[slug]["decisions_needed"]) == count
        assert stewards[slug]["execution"]["state"] == "waiting_for_you"
    assert [item["title"] for item in stewards[second["slug"]]["needs_attention"]] == [
        "Retry the placeholder upload",
    ]

    # Pinned to one board, the worker still reaches the root.
    assert kanban_db.kanban_home() == root
    assert kanban_db.register_db_path() == root / "kanban" / "board_register.db"
    entry = kanban_db.get_register_entry(second["board"])
    assert entry is not None
    assert entry.lifecycle is kanban_db.BoardLifecycle.LIVE
