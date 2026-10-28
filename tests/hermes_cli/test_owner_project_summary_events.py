"""Card B of the Project summary writer plan: Raphael refreshes a Project's summary after material events.

The summary tick reads three sources through the cursors of the Project's state row: the task events of the
Project's own board, its merged delivery rows and the release ledger. One table sorts every event into material,
urgent or ignored. Material events are rewritten at most once an hour, a decision, a merge and a release at once,
and ten quick changes are one rewrite (owner decision of 2026-10-27). Card A's owner rules 1 to 3 hold for every
rewrite request: the kernel's time box, one open task found by one check first, and a failure limit that only delays
the next request by a day.

Each test imports the modules itself, so each one fails on its own before the change.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import os
import sqlite3
import threading
import urllib.parse
from pathlib import Path

import pytest

from hermes_cli import kanban_db, kanban_risk_tier, owner_workspace as ow, projects_db, release_ledger
from hermes_cli.owner_text import check_owner_text
from plugins.dashboard_auth.raphael_workspace import model_policy

PLANNER = "raphael-planner"
BOARD = "project-summaries"
TITLE = "Write the Project summary: "
HOUR = 3600
DAY = 24 * HOUR
T0 = 1_900_000_000

OUTCOME = "The room is booked for Tuesday at noon and the invitations are being written."
NEXT_ACTION = "Raphael sends the agenda to the team on Monday."
DECISION = "Please choose between the large room and the small room."

REGISTRY_FILE = "projects.db"
SUMMARY_FILE = f"kanban/boards/{BOARD}/kanban.db"
LEDGER_FILE = "kanban/release_ledger.db"


def _ops():
    from hermes_cli import owner_project_summary

    return owner_project_summary


def _events():
    from hermes_cli import owner_project_summary_events

    return owner_project_summary_events


class _ProcessDied(BaseException):
    """A gateway restart as the interrupted code sees it: nothing after this point runs."""


@pytest.fixture
def root(tmp_path, monkeypatch):
    """The shared root, entered as the default profile."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(
        model_policy, "configured_assignment_for",
        lambda profile: model_policy.assignment_for(profile, "anthropic"),
    )
    return home


@pytest.fixture
def owner(root):
    return ow.OwnerContext(actor="default", profile="default", session="run_owt")


def _clean(*texts: str) -> None:
    for text in texts:
        assert check_owner_text(text, "planner_prose") == text


def _valid(**changes) -> dict:
    summary = {"outcome": OUTCOME, "next_action": NEXT_ACTION, "owner_decision": DECISION}
    return {"project_summary": {**summary, **changes}}


def _rows(path: Path, sql: str, params=()) -> list:
    conn = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    try:
        return conn.execute(sql, params).fetchall()
    finally:
        conn.close()


def _board_db(root: Path, slug: str) -> Path:
    return root / "kanban" / "boards" / slug / "kanban.db"


@contextlib.contextmanager
def _board(root: Path, slug: str = BOARD):
    with kanban_db.connect_closing(_board_db(root, slug), board=slug) as conn:
        yield conn


def _project(owner, name: str) -> dict:
    """An owner Project with its own board, as the owner kernel creates it."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(ow, "_confirm", lambda *_args, **_kwargs: {"approved": True})
        project = ow.bootstrap(
            owner, idempotency_key="setup-" + name.lower().replace(" ", "-"), name=name,
        )
    assert project["ok"] is True
    project["name"] = name
    return project


def _registry_project(root: Path, name: str, **columns) -> str:
    with projects_db.connect_closing(root / REGISTRY_FILE) as conn:
        return projects_db.create_project(conn, name=name, **columns)


def _summary_of(root: Path, project_id: str):
    with projects_db.connect_closing(root / REGISTRY_FILE) as conn:
        return projects_db.get_project_summary(conn, project_id)


_WHOLE_STATE = (
    "event_cursor", "delivery_cursor", "release_cursor", "pending", "pending_urgent",
    "open_task_id", "last_requested_at", "failures", "updated_at",
)


def _state(root: Path, project_id: str) -> dict | None:
    rows = _rows(
        root / REGISTRY_FILE,
        f"SELECT {', '.join(_WHOLE_STATE)} FROM project_summary_state WHERE project_id = ?",
        (project_id,),
    )
    return dict(zip(_WHOLE_STATE, rows[0])) if rows else None


def _summary_tasks(root: Path, *, archived: bool = False) -> list:
    if not _board_db(root, BOARD).exists():
        return []
    with _board(root) as conn:
        ids = [
            row[0] for row in conn.execute(
                "SELECT id FROM tasks WHERE task_kind = 'work' ORDER BY created_at, rowid"
            )
        ]
        tasks = [kanban_db.get_task(conn, task_id) for task_id in ids]
    return [task for task in tasks if archived or task.status != "archived"]


def _tasks_of(root: Path, project: dict, *, archived: bool = False) -> list:
    return [
        task for task in _summary_tasks(root, archived=archived) if task.title == TITLE + project["name"]
    ]


def _open_task_ids(root: Path, project: dict) -> list:
    """The summary tasks of one Project that are neither done nor archived, counted on a separate connection."""
    return [
        row[0]
        for row in _rows(
            _board_db(root, BOARD),
            "SELECT id FROM tasks WHERE title = ? AND status NOT IN ('done', 'archived') ORDER BY created_at, id",
            (TITLE + project["name"],),
        )
    ]


def _finish(root: Path, task_id: str, metadata: dict | None) -> None:
    """What a dispatched worker does: claim the card, then complete it with metadata."""
    with _board(root) as conn:
        assert kanban_db.claim_task(conn, task_id) is not None
        assert kanban_db.complete_task(
            conn, task_id, summary="Wrote the summary.", metadata=metadata,
        )


def _fail(root: Path, task_id: str) -> None:
    """A summary step that ends without a result: the worker blocks the card."""
    with _board(root) as conn:
        assert kanban_db.claim_task(conn, task_id) is not None
        assert kanban_db.block_task(conn, task_id, reason="The summary could not be written.")


def _task_status(root: Path, task_id: str) -> str:
    return _rows(_board_db(root, BOARD), "SELECT status FROM tasks WHERE id = ?", (task_id,))[0][0]


def _leave(root: Path, project_id: str, how: str) -> None:
    assert how in ("archived", "removal_completed")
    with projects_db.connect_closing(root / REGISTRY_FILE) as registry:
        if how == "archived":
            assert projects_db.archive_project(registry, project_id)
        else:
            assert projects_db.record_removal_operation(
                registry, project_id=project_id, idempotency_key="remove-" + project_id, action="start",
                phase="done", mode="permanent",
            )
            assert projects_db.get_active_removal_operation(registry, project_id) is None


def _order(root: Path, *projects: dict) -> None:
    """Make the tick meet the Projects in this order: the registry lists them oldest first."""
    with projects_db.connect_closing(root / REGISTRY_FILE) as registry, kanban_db.write_txn(registry):
        for age, project in enumerate(projects):
            registry.execute("UPDATE projects SET created_at = ? WHERE id = ?", (1_000 + age, project["project_id"]))


def _listed(root: Path, project: dict):
    with projects_db.connect_closing(root / REGISTRY_FILE) as registry:
        [found] = [one for one in projects_db.list_projects(registry) if one.id == project["project_id"]]
    return found


# The events a Project's board gets, each written by the kernel's own writer, never by a row of SQL.


def _card(
    root: Path, project: dict, title: str = "A card", *, triage: bool = False, assignee: str = "worker",
) -> str:
    with _board(root, project["board"]) as conn:
        return kanban_db.create_task(
            conn, title=title, assignee=assignee, triage=triage, project_id=project["project_id"],
        )


def _complete(root: Path, project: dict, card: str) -> None:
    with _board(root, project["board"]) as conn:
        assert kanban_db.claim_task(conn, card) is not None
        assert kanban_db.complete_task(conn, card, summary="The card is done.", metadata=None)


def _block(root: Path, project: dict, card: str, kind: str) -> None:
    with _board(root, project["board"]) as conn:
        assert kanban_db.block_task(conn, card, reason="Which room should it be?", kind=kind)


def _unblock(root: Path, project: dict, card: str) -> None:
    with _board(root, project["board"]) as conn:
        assert kanban_db.unblock_task(conn, card)


def _review(root: Path, project: dict, card: str) -> None:
    with _board(root, project["board"]) as conn:
        claimed = kanban_db.claim_task(conn, card)
        assert claimed is not None
        kanban_db._set_worker_pid(conn, card, os.getpid())
        assert kanban_db.request_review(
            conn, card, summary="Ready for review.", reviewer="reviewer",
            expected_run_id=claimed.current_run_id,
        )


def _archive_card(root: Path, project: dict, card: str) -> None:
    with _board(root, project["board"]) as conn:
        assert kanban_db.archive_task(conn, card)


def _link(root: Path, project: dict, parent: str, child: str) -> None:
    with _board(root, project["board"]) as conn:
        kanban_db.link_tasks(conn, parent, child)


def _unlink(root: Path, project: dict, parent: str, child: str) -> None:
    with _board(root, project["board"]) as conn:
        assert kanban_db.unlink_tasks(conn, parent, child)


def _decompose(root: Path, project: dict, triage: str) -> None:
    with _board(root, project["board"]) as conn:
        assert kanban_db.decompose_triage_task(
            conn, triage, root_assignee="worker", children=[{"title": "A child"}],
        )


def _quiet(root: Path, project: dict, card: str) -> None:
    """The events that are never material: a claim, a spawn, a heartbeat and a comment."""
    with _board(root, project["board"]) as conn:
        assert kanban_db.claim_task(conn, card) is not None
        kanban_db._set_worker_pid(conn, card, os.getpid())
        assert kanban_db.heartbeat_worker(conn, card, note="Still working.")
        kanban_db.add_comment(conn, card, "owner", "A note for the card.")


def _heartbeat(root: Path, project: dict, card: str) -> None:
    with _board(root, project["board"]) as conn:
        assert kanban_db.heartbeat_worker(conn, card, note="Still working.")


def _kinds(root: Path, project: dict, after: int = 0) -> list:
    """The kinds of the board's events past ``after``, taken with a plain connection."""
    return [
        row[0] for row in _rows(
            _board_db(root, project["board"]), "SELECT kind FROM task_events WHERE id > ? ORDER BY id", (after,),
        )
    ]


def _top(root: Path, project: dict) -> int:
    return _rows(_board_db(root, project["board"]), "SELECT COALESCE(MAX(id), 0) FROM task_events")[0][0]


def _sha(label: str) -> str:
    return hashlib.sha1(label.encode()).hexdigest()


def _delivery(root: Path, project: dict, card: str, state: str | None) -> int:
    """A delivery row as the delivery step leaves it, in the state ``state``: a plain row, since only its state counts."""
    with _board(root, project["board"]) as conn, kanban_db.write_txn(conn):
        return conn.execute(
            "INSERT INTO kanban_deliveries (source_task_id, source_head, created_at, pull_request_state) "
            "VALUES (?, ?, ?, ?)",
            (card, _sha(f"{card}-{state}-{T0}"), T0, state),
        ).lastrowid


def _set_delivery(root: Path, project: dict, delivery: int, state: str) -> None:
    with _board(root, project["board"]) as conn, kanban_db.write_txn(conn):
        conn.execute("UPDATE kanban_deliveries SET pull_request_state = ? WHERE id = ?", (state, delivery))


class _Ledger:
    """The release ledger of the kanban root, written only through its own writers."""

    def __init__(self):
        self.pulls = 0

    def merge(self, label: str, *, card: str, board: str = "") -> dict:
        self.pulls += 1
        with contextlib.closing(release_ledger.connect()) as conn:
            return release_ledger.record_merge(
                conn,
                merge_commit=_sha(label),
                pr_url=f"https://github.com/mtbitcr/hermes-agent/pull/{self.pulls}",
                reviewed_base=_sha(label + "-base"),
                reviewed_head=_sha(label + "-head"),
                reviewed_tree=_sha(label + "-tree"),
                tier=1,
                card_id=card,
                board=board,
                on_main=True,
            )["batch"]

    def accept(self, batch: dict) -> dict:
        with contextlib.closing(release_ledger.connect()) as conn:
            return release_ledger.decide_release(
                conn, batch["batch_id"], decision="accepted", shown_digest=batch["digest"],
                expected_version=batch["version"], decision_ref="decisions-page:release",
            )

    def begin(self, batch: dict) -> None:
        with contextlib.closing(release_ledger.connect()) as conn:
            release_ledger.begin_release(
                conn, batch["batch_id"], prev=release_ledger.last_released(conn) or _sha("live"),
                new=batch["members"][-1]["merge_commit"],
            )

    def recover(self, batch: dict) -> None:
        with contextlib.closing(release_ledger.connect()) as conn:
            release_ledger.begin_recovery(conn, batch["batch_id"])

    def finish(self, batch: dict, outcome: str) -> None:
        with contextlib.closing(release_ledger.connect()) as conn:
            release_ledger.finish_release(conn, batch["batch_id"], outcome=outcome)

    def release(self, label: str, *, card: str, board: str = "") -> dict:
        """Merge one change, accept its batch, begin its release and finish it as released."""
        batch = self.accept(self.merge(label, card=card, board=board))
        self.begin(batch)
        self.finish(batch, "released")
        return batch


def _settle(root: Path, at: int = T0) -> None:
    """Take every Project through its first summary: asked for at ``at``, stored ten seconds later.

    A tick asks for three at most, so a fourth Project is asked for, and stored, ten seconds after the rest.
    """
    ops = _ops()
    _clean(OUTCOME, NEXT_ACTION, DECISION)
    ops.summary_tick(now=at)
    step = 0
    while _summary_tasks(root):
        step += 10
        assert step <= 20
        for task in _summary_tasks(root):
            _finish(root, task.id, _valid())
        ops.summary_tick(now=at + step)
    registry = root / REGISTRY_FILE
    assert _rows(registry, "SELECT count(*) FROM project_summaries") == _rows(
        registry, "SELECT count(*) FROM projects WHERE archived = 0"
    )


def _assert_rewrite(root: Path, project: dict, *, asked_at: int, failures: int = 0) -> str:
    """Card A's owner rules on the one open rewrite task of the Project; returns its id.

    The time box is the kernel's pin, the task is the Project's only open one, and the state row records it
    with the ``failures`` the request was made after.
    """
    [task] = [task for task in _tasks_of(root, project) if task.status == "ready"]
    assert task.model_policy_lock
    kind = kanban_risk_tier.card_work_kind(PLANNER, [], "routine", False)
    assert _rows(
        _board_db(root, BOARD), "SELECT max_runtime_seconds FROM tasks WHERE id = ?", (task.id,),
    ) == [(kanban_risk_tier.pinned_time_box_seconds(kind),)]
    assert (task.assignee, task.created_by, task.project_id) == (PLANNER, "project-summary", None)
    assert task.idempotency_key.startswith(f"project-summary:{project['project_id']}:")
    assert _open_task_ids(root, project) == [task.id]
    state = _state(root, project["project_id"])
    assert (state["open_task_id"], state["last_requested_at"], state["failures"]) == (task.id, asked_at, failures)
    assert (state["pending"], state["pending_urgent"]) == (0, 0)
    return task.id


class _Trace:
    """Every file a tick opens, and every statement it runs on it in order: a spy on ``sqlite3.connect``."""

    def __init__(self, monkeypatch, root: Path):
        self.root = root.resolve()
        self.on = False
        self.opened: list[str] = []
        self.statements: list[tuple[str, str]] = []
        connect = sqlite3.connect

        def spy(database, *args, **kwargs):
            name = self._label(database) if self.on else None
            if name is not None:
                self.opened.append(name)
            conn = connect(database, *args, **kwargs)
            if name is not None:
                conn.set_trace_callback(lambda sql: self.statements.append((name, sql)))
            return conn

        monkeypatch.setattr(sqlite3, "connect", spy)

    def _label(self, database) -> str:
        text = str(database)
        if text.startswith("file:"):
            text = urllib.parse.unquote(urllib.parse.urlparse(text).path)
        path = Path(text).resolve()
        with contextlib.suppress(ValueError):
            return path.relative_to(self.root).as_posix()
        return path.as_posix()

    def __enter__(self) -> "_Trace":
        self.opened.clear()
        self.statements.clear()
        self.on = True
        return self

    def __exit__(self, *_exc) -> None:
        self.on = False

    def board(self, project: dict) -> str:
        return f"kanban/boards/{project['board']}/kanban.db"


def _assert_the_check_comes_first(trace: _Trace) -> None:
    """The first look-up of a task by its key is on the summary board, and only the registry was read before it."""
    statements = trace.statements
    check = next(
        index for index, (name, sql) in enumerate(statements)
        if name == SUMMARY_FILE and "substr(idempotency_key" in sql
    )
    assert {name for name, _sql in statements[:check]} <= {REGISTRY_FILE, SUMMARY_FILE}
    assert {name for name, _sql in statements[check:]} - {REGISTRY_FILE, SUMMARY_FILE}


def _garble(path: Path) -> None:
    """Make every read of the file fail: no journal beside it, and bytes that are no database."""
    for suffix in ("-wal", "-shm"):
        Path(str(path) + suffix).unlink(missing_ok=True)
    path.write_bytes(b"this is not a database " * 400)


class _Reads:
    """The two reads a tick makes before it publishes: counts them, and refuses the one named in ``failing``."""

    def __init__(self, monkeypatch):
        self.calls: list = []
        self.failing: str | None = None
        route, snapshot = ow._resolved_route_pin, ow.project_steward_snapshot

        def read_route(*args, **kwargs):
            self.calls.append(("route", None))
            if self.failing == "route":
                raise ow.OwnerWorkspaceError("invalid_model_route", "the planning role has no usable route")
            return route(*args, **kwargs)

        def read_snapshot(**kwargs):
            self.calls.append(("snapshot", kwargs.get("project_id")))
            if self.failing == "snapshot":
                raise ow.OwnerWorkspaceError("snapshot_unavailable", "the Project board is unavailable")
            return snapshot(**kwargs)

        monkeypatch.setattr(ow, "_resolved_route_pin", read_route)
        monkeypatch.setattr(ow, "project_steward_snapshot", read_snapshot)


def _crash_after_the_task_is_made(ops, monkeypatch, now: int) -> None:
    """Interrupt a tick right after create_task commits and before the state row records the task."""
    create_task = kanban_db.create_task

    def create_then_die(*args, **kwargs):
        create_task(*args, **kwargs)
        raise _ProcessDied

    monkeypatch.setattr(kanban_db, "create_task", create_then_die)
    try:
        with pytest.raises(_ProcessDied):
            ops.summary_tick(now=now)
    finally:
        monkeypatch.setattr(kanban_db, "create_task", create_task)


def _enter_planning_worker(monkeypatch, root: Path, slug: str, task_id: str) -> None:
    """Take on the store pins kanban_db_dispatch._default_spawn gives a worker.

    The board pins are resolved first, in the dispatcher's own environment.
    """
    pins = {
        "HERMES_KANBAN_DB": str(kanban_db.kanban_db_path(board=slug)),
        "HERMES_KANBAN_WORKSPACES_ROOT": str(kanban_db.workspaces_root(board=slug)),
        "HERMES_KANBAN_BOARD": slug,
    }
    monkeypatch.setenv("HERMES_HOME", str(root / "profiles" / PLANNER))
    monkeypatch.setenv("HERMES_PROFILE", PLANNER)
    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    for name, value in pins.items():
        monkeypatch.setenv(name, value)


REWRITTEN = "The garden beds are planted and the watering plan is set."


def _store_rewrite(root: Path, task_id: str, tick_at: int, outcome: str = REWRITTEN) -> None:
    """The worker finishes the rewrite task, and the next tick stores what it wrote."""
    _finish(root, task_id, _valid(outcome=outcome))
    _ops().summary_tick(now=tick_at)


# Test 1: an ordinary material event waits for the hour, then it is one rewrite.


def test_a_finish_inside_the_hour_gives_no_task_and_after_the_hour_exactly_one(root, owner, monkeypatch):
    ops = _ops()
    _clean(OUTCOME, NEXT_ACTION, DECISION, REWRITTEN)
    garden = _project(owner, "Garden Plan")
    pid = garden["project_id"]
    card = _card(root, garden)
    _settle(root)
    settled = _state(root, pid)
    assert (settled["last_requested_at"], settled["pending"], settled["failures"]) == (T0, 0, 0)
    assert _summary_of(root, pid)["revision"] == 1

    _complete(root, garden, card)

    assert ops.summary_tick(now=T0 + HOUR - 1) == 0
    assert _tasks_of(root, garden) == []
    waiting = _state(root, pid)
    assert (waiting["pending"], waiting["pending_urgent"], waiting["open_task_id"]) == (1, 0, None)
    assert waiting["event_cursor"] == _top(root, garden)
    kept = ("last_requested_at", "failures", "updated_at")
    assert {key: waiting[key] for key in kept} == {key: settled[key] for key in kept}

    with _Trace(monkeypatch, root) as trace:
        assert ops.summary_tick(now=T0 + HOUR) == 1
    rewrite = _assert_rewrite(root, garden, asked_at=T0 + HOUR)
    _assert_the_check_comes_first(trace)
    assert trace.board(garden) in trace.opened

    for later in (T0 + HOUR + 1, T0 + 5 * HOUR):
        assert ops.summary_tick(now=later) == 0
    assert _open_task_ids(root, garden) == [rewrite]
    _store_rewrite(root, rewrite, T0 + 5 * HOUR + 10)
    stored = _summary_of(root, pid)
    assert (stored["revision"], stored["outcome"]) == (2, REWRITTEN)
    assert _tasks_of(root, garden) == []


# Test 2: ten quick changes are one rewrite, and the hour runs from that request.


def test_ten_quick_changes_give_one_task(root, owner, monkeypatch):
    ops = _ops()
    _clean(OUTCOME, NEXT_ACTION, DECISION, REWRITTEN)
    garden = _project(owner, "Garden Plan")
    pid = garden["project_id"]
    steps = [_card(root, garden, f"Step {number}") for number in range(5)]
    _settle(root)
    asked = len(_tasks_of(root, garden, archived=True))
    before = _top(root, garden)

    at = T0 + 100
    for step in steps:
        _complete(root, garden, step)
        at += 30
        assert ops.summary_tick(now=at) == 0
        _card(root, garden, "Another step")
        at += 30
        assert ops.summary_tick(now=at) == 0
    made = _kinds(root, garden, after=before)
    assert made.count("completed") + made.count("created") == 10
    assert len(_tasks_of(root, garden, archived=True)) == asked
    waiting = _state(root, pid)
    assert (waiting["pending"], waiting["pending_urgent"]) == (1, 0)

    assert ops.summary_tick(now=T0 + HOUR) == 1
    rewrite = _assert_rewrite(root, garden, asked_at=T0 + HOUR)
    assert _state(root, pid)["event_cursor"] == _top(root, garden)
    _store_rewrite(root, rewrite, T0 + HOUR + 10)
    assert _summary_of(root, pid)["revision"] == 2

    assert ops.summary_tick(now=T0 + 9 * HOUR) == 0
    assert len(_tasks_of(root, garden, archived=True)) == asked + 1
    assert _state(root, pid)["pending"] == 0

    _card(root, garden, "One more step")
    assert ops.summary_tick(now=T0 + 9 * HOUR + 60) == 0
    assert ops.summary_tick(now=T0 + 10 * HOUR - 1) == 0
    assert _tasks_of(root, garden) == []
    assert ops.summary_tick(now=T0 + 10 * HOUR) == 1
    _assert_rewrite(root, garden, asked_at=T0 + 10 * HOUR)


# Test 3: a decision asked, or answered, is rewritten at once, inside the hour.


def test_a_needs_input_block_and_its_answer_each_give_a_task_inside_the_hour(root, owner, monkeypatch):
    ops = _ops()
    _clean(OUTCOME, NEXT_ACTION, DECISION, REWRITTEN)
    garden = _project(owner, "Garden Plan")
    pid = garden["project_id"]
    asking = _card(root, garden, "Choose the room")
    other = _card(root, garden, "Book the room")
    _settle(root)

    _block(root, garden, other, "capability")
    assert ops.summary_tick(now=T0 + 100) == 0
    assert _tasks_of(root, garden) == []
    assert (_state(root, pid)["pending"], _state(root, pid)["pending_urgent"]) == (1, 0)

    _block(root, garden, asking, "needs_input")
    with _Trace(monkeypatch, root) as trace:
        assert ops.summary_tick(now=T0 + 200) == 1
    first = _assert_rewrite(root, garden, asked_at=T0 + 200)
    _assert_the_check_comes_first(trace)
    _store_rewrite(root, first, T0 + 210)
    assert _summary_of(root, pid)["revision"] == 2
    assert _tasks_of(root, garden) == []

    _unblock(root, garden, asking)
    assert ops.summary_tick(now=T0 + 300) == 1
    second = _assert_rewrite(root, garden, asked_at=T0 + 300)
    assert second != first
    _store_rewrite(root, second, T0 + 310)
    assert _summary_of(root, pid)["revision"] == 3

    _unblock(root, garden, other)
    assert ops.summary_tick(now=T0 + 330) == 0
    assert _tasks_of(root, garden) == []
    assert (_state(root, pid)["pending"], _state(root, pid)["pending_urgent"]) == (1, 0)
    assert ops.summary_tick(now=T0 + 300 + HOUR - 1) == 0
    assert ops.summary_tick(now=T0 + 300 + HOUR) == 1
    _assert_rewrite(root, garden, asked_at=T0 + 300 + HOUR)


def _append(root: Path, project: dict, card: str, kind: str, payload: dict | None = None) -> None:
    """An event the kernel writes next to other work, here on its own: ``delivery_merged`` and the like."""
    with _board(root, project["board"]) as conn, kanban_db.write_txn(conn):
        kanban_db._append_event(conn, card, kind, payload)


def _ledger_top(root: Path) -> int:
    return _rows(root / LEDGER_FILE, "SELECT COALESCE(MAX(id), 0) FROM release_events")[0][0]


def _flags(root: Path, project: dict) -> tuple:
    state = _state(root, project["project_id"])
    return (state["pending"], state["pending_urgent"])


# Test 4: a merged delivery row, and a released batch that holds a card of the Project, are rewritten at once.


def test_a_merged_delivery_row_gives_a_task_at_once(root, owner, monkeypatch):
    from hermes_cli import kanban_delivery

    assert kanban_delivery._MERGED == "merged"
    ops = _ops()
    _clean(OUTCOME, NEXT_ACTION, DECISION, REWRITTEN)
    garden = _project(owner, "Garden Plan")
    desk = _project(owner, "Planning Desk")
    pid = garden["project_id"]
    unmerged = [
        _delivery(root, garden, _card(root, garden, f"Card {number}"), state)
        for number, state in enumerate((None, "open", kanban_delivery._ARMED, kanban_delivery._CLOSED_OUTSIDE))
    ]
    _settle(root)
    assert _state(root, pid)["delivery_cursor"] == 0

    assert ops.summary_tick(now=T0 + 100) == 0
    assert _flags(root, garden) == (0, 0)
    assert _state(root, pid)["delivery_cursor"] == 0

    _set_delivery(root, garden, unmerged[1], "merged")
    with _Trace(monkeypatch, root) as trace:
        assert ops.summary_tick(now=T0 + 200) == 1
    first = _assert_rewrite(root, garden, asked_at=T0 + 200)
    _assert_the_check_comes_first(trace)
    assert _state(root, pid)["delivery_cursor"] == unmerged[1]
    assert _tasks_of(root, desk) == []

    _store_rewrite(root, first, T0 + 210)
    assert ops.summary_tick(now=T0 + 220) == 0
    assert _tasks_of(root, garden) == []
    later = _delivery(root, garden, _card(root, garden, "A late card"), "merged")
    assert later > unmerged[1]
    assert ops.summary_tick(now=T0 + 300) == 1
    _assert_rewrite(root, garden, asked_at=T0 + 300)
    assert _state(root, pid)["delivery_cursor"] == later


def test_a_released_batch_that_holds_a_card_of_the_project_gives_a_task_at_once(root, owner, monkeypatch):
    ops = _ops()
    _clean(OUTCOME, NEXT_ACTION, DECISION, REWRITTEN)
    garden = _project(owner, "Garden Plan")
    desk = _project(owner, "Planning Desk")
    pid = garden["project_id"]
    beds, lamp = _card(root, garden, "Plant the beds"), _card(root, desk, "Order the lamp")
    _settle(root)
    assert not (root / LEDGER_FILE).exists()
    assert _state(root, pid)["release_cursor"] == 0
    ledger = _Ledger()

    batch = ledger.accept(ledger.merge("beds", card=beds, board=garden["board"]))
    at = T0 + 100
    for step in (lambda: None, lambda: ledger.begin(batch), lambda: ledger.finish(batch, "failed"),
                 lambda: ledger.recover(batch)):
        step()
        at += 10
        assert ops.summary_tick(now=at) == 0
        assert _tasks_of(root, garden) == [] and _tasks_of(root, desk) == []
        assert _flags(root, garden) == (0, 0)
    assert _state(root, pid)["release_cursor"] == _ledger_top(root) > 0

    ledger.finish(batch, "released")
    with _Trace(monkeypatch, root) as trace:
        assert ops.summary_tick(now=at + 10) == 1
    first = _assert_rewrite(root, garden, asked_at=at + 10)
    _assert_the_check_comes_first(trace)
    assert LEDGER_FILE in trace.opened
    assert _state(root, pid)["release_cursor"] == _ledger_top(root)
    assert _tasks_of(root, desk) == []
    _store_rewrite(root, first, at + 20)
    assert ops.summary_tick(now=at + 30) == 0
    assert _tasks_of(root, garden) == []

    ledger.release("lamp", card=lamp, board=desk["board"])
    assert ops.summary_tick(now=at + 40) == 1
    assert [task.title for task in _summary_tasks(root)] == [TITLE + "Planning Desk"]
    _assert_rewrite(root, desk, asked_at=at + 40)
    assert _tasks_of(root, garden) == []


def test_a_release_is_matched_by_the_board_slug_or_by_the_card_and_never_by_another_board(root, owner):
    ops = _ops()
    _clean(OUTCOME, NEXT_ACTION, DECISION, REWRITTEN)
    garden = _project(owner, "Garden Plan")
    pid = garden["project_id"]
    water = _card(root, garden, "Water the beds")
    _settle(root)
    ledger = _Ledger()

    ledger.release("stranger", card="t_nobody", board="elsewhere")
    assert ops.summary_tick(now=T0 + 100) == 0
    assert _tasks_of(root, garden) == []
    assert _state(root, pid)["release_cursor"] == _ledger_top(root) > 0

    ledger.release("named", card="t_nobody", board=garden["board"])
    assert ops.summary_tick(now=T0 + 200) == 1
    named = _assert_rewrite(root, garden, asked_at=T0 + 200)
    _store_rewrite(root, named, T0 + 210)

    waiting = ledger.merge("manual", card=water)
    assert waiting["members"][-1]["board"] == ""
    with contextlib.closing(release_ledger.connect()) as conn:
        outside = release_ledger.settle_outside_release(
            conn, running=waiting["members"][-1]["merge_commit"], contains=lambda _merge, _running: True,
        )
    assert outside["released"] is not None
    assert ops.summary_tick(now=T0 + 300) == 1
    _assert_rewrite(root, garden, asked_at=T0 + 300)


# Test 5: heartbeat, comment, claim and every event on the summary board give no task.


def test_heartbeat_comment_claim_and_unlisted_kinds_give_no_task_while_a_finish_does(root, owner):
    ops = _ops()
    _clean(OUTCOME, NEXT_ACTION, DECISION, REWRITTEN)
    garden = _project(owner, "Garden Plan")
    pid = garden["project_id"]
    busy, plain = _card(root, garden, "Busy card"), _card(root, garden, "Plain card")
    _settle(root)
    before = _top(root, garden)

    _quiet(root, garden, busy)
    for kind in ("promoted", "delivery_merged", "dependency_wait", "block_loop_detected", "owner_retry"):
        _append(root, garden, busy, kind, {"note": "written next to other work"})
    made = set(_kinds(root, garden, after=before))
    assert {"claimed", "spawned", "heartbeat", "commented", "promoted", "delivery_merged"} <= made
    assert not made & {"created", "completed", "blocked", "unblocked", "review_requested"}

    for at in (T0 + 100, T0 + 2 * HOUR, T0 + 3 * HOUR):
        assert ops.summary_tick(now=at) == 0
    assert _tasks_of(root, garden) == []
    state = _state(root, pid)
    assert (state["pending"], state["pending_urgent"]) == (0, 0)
    assert state["event_cursor"] == _top(root, garden)

    _complete(root, garden, plain)
    assert ops.summary_tick(now=T0 + 3 * HOUR + 10) == 1
    _assert_rewrite(root, garden, asked_at=T0 + 3 * HOUR + 10)


def test_events_on_the_summary_board_and_projects_without_a_board_give_no_task(root, owner):
    ops = _ops()
    _clean(OUTCOME, NEXT_ACTION, DECISION, REWRITTEN)
    garden = _project(owner, "Garden Plan")
    plain = _card(root, garden, "Plain card")
    summarised = {}
    for name, columns in (("Loop", {"board_slug": BOARD}), ("Bare", {})):
        summarised[name] = _registry_project(root, name, **columns)
        with projects_db.connect_closing(root / REGISTRY_FILE) as registry:
            projects_db.write_project_summary(registry, summarised[name], outcome=OUTCOME, next_action=NEXT_ACTION)
    _settle(root)
    on_the_summary_board = set(_kinds(root, {"board": BOARD}))
    assert {"created", "claimed", "completed"} <= on_the_summary_board

    for at in (T0 + 100, T0 + 2 * HOUR):
        assert ops.summary_tick(now=at) == 0
    for name, project_id in summarised.items():
        assert _tasks_of(root, {"name": name}, archived=True) == [], name
        state = _state(root, project_id)
        assert (state["event_cursor"], state["pending"], state["open_task_id"], state["failures"]) == (0, 0, None, 0)

    _complete(root, garden, plain)
    assert ops.summary_tick(now=T0 + 2 * HOUR + 10) == 1
    _assert_rewrite(root, garden, asked_at=T0 + 2 * HOUR + 10)
    assert _tasks_of(root, {"name": "Loop"}, archived=True) == []


# Test 6: a link, an unlink and an archive are material but not urgent; so is every other graph and progress event.

_CHANGES = {
    "created": lambda root, project, cards: _card(root, project, "A new card"),
    "linked": lambda root, project, cards: _link(root, project, cards["loose_parent"], cards["loose_child"]),
    "unlinked": lambda root, project, cards: _unlink(root, project, cards["tied_parent"], cards["tied_child"]),
    "archived": lambda root, project, cards: _archive_card(root, project, cards["spare"]),
    "decomposed": lambda root, project, cards: _decompose(root, project, cards["triage"]),
    "completed": lambda root, project, cards: _complete(root, project, cards["worker"]),
    "blocked": lambda root, project, cards: _block(root, project, cards["worker"], "capability"),
    "unblocked": lambda root, project, cards: _unblock(root, project, cards["held"]),
}


@pytest.mark.parametrize("kind", sorted(_CHANGES))
def test_graph_progress_and_blocker_events_are_material_but_not_urgent(root, owner, kind):
    ops = _ops()
    _clean(OUTCOME, NEXT_ACTION, DECISION, REWRITTEN)
    garden = _project(owner, "Garden Plan")
    cards = {
        name: _card(root, garden, name, triage=name == "triage")
        for name in ("loose_parent", "loose_child", "tied_parent", "tied_child", "spare", "triage", "worker", "held")
    }
    _link(root, garden, cards["tied_parent"], cards["tied_child"])
    _block(root, garden, cards["held"], "capability")
    _settle(root)
    before = _top(root, garden)

    _CHANGES[kind](root, garden, cards)

    assert kind in _kinds(root, garden, after=before)
    assert ops.summary_tick(now=T0 + 100) == 0
    assert _tasks_of(root, garden) == []
    assert _flags(root, garden) == (1, 0)
    assert ops.summary_tick(now=T0 + HOUR - 1) == 0
    assert ops.summary_tick(now=T0 + HOUR) == 1
    _assert_rewrite(root, garden, asked_at=T0 + HOUR)


# Test 7: an urgent event that comes while a task is open waits for the harvest, then is asked for at once.


def test_an_urgent_event_while_a_task_is_open_makes_no_second_task_and_is_asked_for_after_the_harvest(
    root, owner, monkeypatch,
):
    ops = _ops()
    _clean(OUTCOME, NEXT_ACTION, DECISION, REWRITTEN)
    garden = _project(owner, "Garden Plan")
    pid = garden["project_id"]
    booking, asking = _card(root, garden, "Book the room"), _card(root, garden, "Choose the room")
    _settle(root)

    _complete(root, garden, booking)
    assert ops.summary_tick(now=T0 + HOUR) == 1
    first = _assert_rewrite(root, garden, asked_at=T0 + HOUR)
    asked = len(_tasks_of(root, garden, archived=True))
    open_state = _state(root, pid)
    assert open_state["event_cursor"] == _top(root, garden)

    _block(root, garden, asking, "needs_input")
    assert _top(root, garden) > open_state["event_cursor"]
    for at in (T0 + HOUR + 100, T0 + 3 * HOUR):
        with _Trace(monkeypatch, root) as trace:
            assert ops.summary_tick(now=at) == 0
        assert trace.board(garden) not in trace.opened
        assert LEDGER_FILE not in trace.opened
        assert _open_task_ids(root, garden) == [first]
        assert len(_tasks_of(root, garden, archived=True)) == asked
        assert _state(root, pid) == open_state

    _finish(root, first, _valid(outcome=REWRITTEN))
    assert ops.summary_tick(now=T0 + 3 * HOUR + 10) == 1
    assert _summary_of(root, pid)["revision"] == 2
    second = _assert_rewrite(root, garden, asked_at=T0 + 3 * HOUR + 10)
    assert second != first
    assert _state(root, pid)["event_cursor"] == _top(root, garden)
    _store_rewrite(root, second, T0 + 3 * HOUR + 20)
    assert _summary_of(root, pid)["revision"] == 3
    assert _tasks_of(root, garden) == []


def test_a_task_made_before_a_crash_is_found_first_and_no_event_source_is_read(root, owner, monkeypatch):
    ops = _ops()
    _clean(OUTCOME, NEXT_ACTION, DECISION, REWRITTEN)
    reads = _Reads(monkeypatch)
    garden = _project(owner, "Garden Plan")
    pid = garden["project_id"]
    asking = _card(root, garden, "Choose the room")
    _settle(root)
    _block(root, garden, asking, "needs_input")
    before = _state(root, pid)

    _crash_after_the_task_is_made(ops, monkeypatch, T0 + 100)
    [orphan] = _open_task_ids(root, garden)
    assert _state(root, pid) == before

    reads.failing = "route"
    reads.calls.clear()
    _card(root, garden, "Another card")
    with _Trace(monkeypatch, root) as trace:
        assert ops.summary_tick(now=T0 + 200) == 0
    assert reads.calls == []
    assert trace.board(garden) not in trace.opened
    assert LEDGER_FILE not in trace.opened
    assert _open_task_ids(root, garden) == [orphan]
    adopted = _state(root, pid)
    assert (adopted["open_task_id"], adopted["failures"], adopted["last_requested_at"]) == (orphan, 0, T0 + 200)
    assert adopted["event_cursor"] == before["event_cursor"]


@pytest.mark.parametrize("origin", ["batched", "urgent"])
def test_a_rewrite_request_names_no_time_limit_and_the_kernel_pins_it(root, owner, monkeypatch, origin):
    ops = _ops()
    _clean(OUTCOME, NEXT_ACTION, DECISION, REWRITTEN)
    garden = _project(owner, "Garden Plan")
    booking, asking = _card(root, garden, "Book the room"), _card(root, garden, "Choose the room")
    _settle(root)
    if origin == "batched":
        _complete(root, garden, booking)
        at = T0 + HOUR
    else:
        _block(root, garden, asking, "needs_input")
        at = T0 + 100

    made = []
    create_task = kanban_db.create_task

    def spy(*args, **kwargs):
        made.append(kwargs)
        return create_task(*args, **kwargs)

    monkeypatch.setattr(kanban_db, "create_task", spy)
    assert ops.summary_tick(now=at) == 1
    [kwargs] = made
    assert (kwargs["created_by"], kwargs["assignee"]) == ("project-summary", PLANNER)
    assert "max_runtime_seconds" not in kwargs
    _assert_rewrite(root, garden, asked_at=at)


# Test 9: while failures are counted, an urgent event waits for the backoff, and no turn that asks for nothing shortens it.


def test_an_urgent_event_waits_for_the_backoff_while_failures_are_counted(root, owner):
    ops = _ops()
    _clean(OUTCOME, NEXT_ACTION, DECISION, REWRITTEN)
    garden = _project(owner, "Garden Plan")
    pid = garden["project_id"]
    booking, asking = _card(root, garden, "Book the room"), _card(root, garden, "Choose the room")
    _settle(root)

    _complete(root, garden, booking)
    assert ops.summary_tick(now=T0 + HOUR) == 1
    _fail(root, _assert_rewrite(root, garden, asked_at=T0 + HOUR))
    assert ops.summary_tick(now=T0 + HOUR + 10) == 0
    failed = _state(root, pid)
    assert (failed["failures"], failed["updated_at"], failed["pending"], failed["open_task_id"]) == (
        1, T0 + HOUR + 10, 1, None,
    )

    _block(root, garden, asking, "needs_input")
    due = T0 + 2 * HOUR + 10
    for at in (T0 + HOUR + 100, due - 1):
        assert ops.summary_tick(now=at) == 0
        assert _tasks_of(root, garden) == []
        waiting = _state(root, pid)
        assert (waiting["pending"], waiting["pending_urgent"]) == (1, 1)
        assert waiting["event_cursor"] == _top(root, garden)
        assert (waiting["failures"], waiting["updated_at"]) == (1, failed["updated_at"])
    assert ops.summary_tick(now=due) == 1
    _assert_rewrite(root, garden, asked_at=due, failures=1)


def _ask_and_fail(root: Path, project: dict, at: int, *, failures: int) -> int:
    """The request due at ``at`` is made and its task fails; the tick ten seconds later counts the failure.

    Returns the clock of that tick, the moment the backoff runs from.
    """
    ops = _ops()
    assert ops.summary_tick(now=at) == 1
    _fail(root, _assert_rewrite(root, project, asked_at=at, failures=failures))
    assert ops.summary_tick(now=at + 10) == 0
    state = _state(root, project["project_id"])
    assert (state["failures"], state["updated_at"], state["open_task_id"], state["pending"]) == (
        failures + 1, at + 10, None, 1,
    )
    return at + 10


@pytest.mark.parametrize("origin", ["batched", "urgent"])
def test_after_the_failure_limit_the_next_request_waits_a_day_and_no_longer(root, owner, origin):
    ops = _ops()
    _clean(OUTCOME, NEXT_ACTION, DECISION, REWRITTEN)
    garden = _project(owner, "Garden Plan")
    pid = garden["project_id"]
    booking, asking = _card(root, garden, "Book the room"), _card(root, garden, "Choose the room")
    _settle(root)
    if origin == "batched":
        _complete(root, garden, booking)
        first_at = T0 + HOUR
        assert ops.summary_tick(now=first_at - 1) == 0
    else:
        _block(root, garden, asking, "needs_input")
        first_at = T0 + 100

    failed_at = _ask_and_fail(root, garden, first_at, failures=0)
    for failures, wait in ((1, HOUR), (2, 2 * HOUR), (3, DAY), (4, DAY)):
        assert ops.summary_tick(now=failed_at + wait - 1) == 0
        failed_at = _ask_and_fail(root, garden, failed_at + wait, failures=failures)
    assert _state(root, pid)["failures"] == 5

    assert ops.summary_tick(now=failed_at + DAY - 1) == 0
    assert ops.summary_tick(now=failed_at + DAY) == 1
    rewrite = _assert_rewrite(root, garden, asked_at=failed_at + DAY, failures=5)
    _store_rewrite(root, rewrite, failed_at + DAY + 10)
    assert _state(root, pid)["failures"] == 0
    assert _summary_of(root, pid)["revision"] == 2
    assert _tasks_of(root, garden) == []


# Owner rule 1: a failure before the task is published changes the failure count and the clock, and nothing else.


@pytest.mark.parametrize("failing", ["route", "snapshot"])
def test_a_failure_before_the_task_is_made_changes_only_the_failure_count_and_the_clock(
    root, owner, monkeypatch, failing,
):
    ops = _ops()
    _clean(OUTCOME, NEXT_ACTION, DECISION, REWRITTEN)
    reads = _Reads(monkeypatch)
    garden = _project(owner, "Garden Plan")
    pid = garden["project_id"]
    asking = _card(root, garden, "Choose the room")
    _settle(root)
    _block(root, garden, asking, "needs_input")
    before = _state(root, pid)
    asked = len(_tasks_of(root, garden, archived=True))

    reads.failing = failing
    reads.calls.clear()
    assert ops.summary_tick(now=T0 + 100) == 0
    assert [name for name, _project_id in reads.calls][-1:] == [failing]
    after = _state(root, pid)
    assert (after["failures"], after["updated_at"]) == (1, T0 + 100)
    unmoved = [name for name in _WHOLE_STATE if name not in ("failures", "updated_at")]
    assert {name: after[name] for name in unmoved} == {name: before[name] for name in unmoved}
    assert len(_tasks_of(root, garden, archived=True)) == asked

    reads.failing = None
    assert ops.summary_tick(now=T0 + 100 + HOUR - 1) == 0
    assert ops.summary_tick(now=T0 + 100 + HOUR) == 1
    _assert_rewrite(root, garden, asked_at=T0 + 100 + HOUR, failures=1)
    assert _state(root, pid)["event_cursor"] == _top(root, garden)


# Test 8: two ticks at the same time, on two connections, make one task; a tick that loses a swap moves nothing.


def test_two_ticks_at_the_same_time_on_two_connections_make_one_task(root, owner, monkeypatch):
    ops = _ops()
    _clean(OUTCOME, NEXT_ACTION, DECISION, REWRITTEN)
    garden = _project(owner, "Garden Plan")
    pid = garden["project_id"]
    asking = _card(root, garden, "Choose the room")
    _settle(root)
    _block(root, garden, asking, "needs_input")
    asked = len(_tasks_of(root, garden, archived=True))

    meeting = threading.Barrier(2, timeout=10)
    one_at_a_time = threading.Lock()
    create_task = kanban_db.create_task

    def create_then_meet(*args, **kwargs):
        """Both ticks hold the same unclaimed row and the same task before either one makes its claim."""
        with one_at_a_time:
            task_id = create_task(*args, **kwargs)
        meeting.wait()
        return task_id

    monkeypatch.setattr(kanban_db, "create_task", create_then_meet)
    results: list = []
    errors: list = []

    def tick() -> None:
        try:
            results.append(ops.summary_tick(now=T0 + 100))
        except BaseException as exc:
            errors.append(exc)
            meeting.abort()

    threads = [threading.Thread(target=tick) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
    assert not errors, errors
    assert not any(thread.is_alive() for thread in threads)

    assert sorted(results) == [0, 1]
    _assert_rewrite(root, garden, asked_at=T0 + 100)
    assert len(_tasks_of(root, garden, archived=True)) == asked + 1
    assert _state(root, pid)["event_cursor"] == _top(root, garden)


def test_a_tick_that_loses_the_swap_counts_nothing_and_moves_nothing(root, owner, monkeypatch):
    ops, events = _ops(), _events()
    _clean(OUTCOME, NEXT_ACTION, DECISION, REWRITTEN)
    garden = _project(owner, "Garden Plan")
    pid = garden["project_id"]
    booking, asking = _card(root, garden, "Book the room"), _card(root, garden, "Choose the room")
    _settle(root)
    _complete(root, garden, booking)

    read = events.read_events
    inner: dict = {"started": False}

    def read_then_lose_the_race(project, cursors):
        batch = read(project, cursors)
        if not inner["started"]:
            inner["started"] = True
            _block(root, garden, asking, "needs_input")
            inner["asked"] = ops.summary_tick(now=T0 + 101)
            inner["state"] = _state(root, pid)
        return batch

    monkeypatch.setattr(events, "read_events", read_then_lose_the_race)
    assert ops.summary_tick(now=T0 + 100) == 0
    assert inner["asked"] == 1
    assert _state(root, pid) == inner["state"]
    _assert_rewrite(root, garden, asked_at=T0 + 101)
    assert _state(root, pid)["event_cursor"] == _top(root, garden)


# The cap of new tasks per tick holds for rewrite requests too.


def test_rewrite_requests_share_the_cap_of_new_tasks_per_tick(root, owner):
    ops = _ops()
    _clean(OUTCOME, NEXT_ACTION, DECISION, REWRITTEN)
    projects = [_project(owner, f"Plan {name}") for name in "ABCD"]
    cards = [_card(root, project, "Choose the room") for project in projects]
    _settle(root)
    assert ops.MAX_NEW_PER_TICK == 3
    for project, card in zip(projects, cards):
        _block(root, project, card, "needs_input")

    assert ops.summary_tick(now=T0 + 100) == 3
    waiting = [project for project in projects if not _tasks_of(root, project)]
    assert len(waiting) == 1
    assert ops.summary_tick(now=T0 + 110) == 1
    [rewrite] = _tasks_of(root, waiting[0])
    assert rewrite.status == "ready"
    assert sum(len(_tasks_of(root, project)) for project in projects) == 4


# Cursors that card A left at zero mean "from the start of the source", never "not yet read".


def _zero_cursors(root: Path, project_id: str) -> None:
    with projects_db.connect_closing(root / REGISTRY_FILE) as registry, kanban_db.write_txn(registry):
        registry.execute(
            "UPDATE project_summary_state SET event_cursor = 0, delivery_cursor = 0, release_cursor = 0 "
            "WHERE project_id = ?",
            (project_id,),
        )


@pytest.mark.parametrize("source", ["events", "deliveries", "ledger"])
def test_cursors_that_card_a_left_at_zero_read_each_source_from_its_start(root, owner, source):
    ops = _ops()
    _clean(OUTCOME, NEXT_ACTION, DECISION, REWRITTEN)
    garden = _project(owner, "Garden Plan")
    pid = garden["project_id"]
    card = _card(root, garden, "Plant the beds")
    if source == "deliveries":
        _delivery(root, garden, card, "merged")
    if source == "ledger":
        _Ledger().release("beds", card=card, board=garden["board"])
    _settle(root)
    _zero_cursors(root, pid)
    assert _state(root, pid)["last_requested_at"] == T0

    at = T0 + 100
    if source == "events":
        assert ops.summary_tick(now=at) == 0
        assert _flags(root, garden) == (1, 0)
        assert _state(root, pid)["event_cursor"] == _top(root, garden) > 0
        assert ops.summary_tick(now=T0 + HOUR - 1) == 0
        at = T0 + HOUR
    assert ops.summary_tick(now=at) == 1
    _assert_rewrite(root, garden, asked_at=at)
    state = _state(root, pid)
    assert state["event_cursor"] == _top(root, garden)
    if source == "deliveries":
        assert state["delivery_cursor"] > 0
    if source == "ledger":
        assert state["release_cursor"] == _ledger_top(root) > 0


# Test 10: a Project that is gone is skipped before any read, with no failure; a live Project in the same tick is asked.


@pytest.mark.parametrize("order", ["gone_first", "live_first"])
@pytest.mark.parametrize("how", ["removal_completed", "archived"])
def test_a_project_that_is_gone_is_skipped_without_a_read_or_a_failure_while_a_live_one_is_asked(
    root, owner, monkeypatch, caplog, how, order,
):
    ops = _ops()
    _clean(OUTCOME, NEXT_ACTION, DECISION, REWRITTEN)
    live, gone = _project(owner, "Garden Plan"), _project(owner, "Old Plan")
    cards = {project["name"]: _card(root, project, "Choose the room") for project in (live, gone)}
    _settle(root)
    for project in (live, gone):
        _block(root, project, cards[project["name"]], "needs_input")
    _leave(root, gone["project_id"], how)
    _garble(_board_db(root, gone["board"]))
    _order(root, *((gone, live) if order == "gone_first" else (live, gone)))
    before = _state(root, gone["project_id"])
    assert (before["pending"], before["failures"], before["open_task_id"]) == (0, 0, None)
    asked = len(_tasks_of(root, gone, archived=True))

    with _Trace(monkeypatch, root) as trace, caplog.at_level(logging.WARNING):
        assert ops.summary_tick(now=T0 + 100) == 1
    _assert_rewrite(root, live, asked_at=T0 + 100)
    assert len(_tasks_of(root, gone, archived=True)) == asked
    assert _state(root, gone["project_id"]) == before
    assert trace.board(gone) not in trace.opened
    assert [record.getMessage() for record in caplog.records if record.levelno >= logging.WARNING] == []


# A source that cannot be read ends that Project's turn with nothing changed; the other Projects go on.


@pytest.mark.parametrize("order", ["broken_first", "broken_second"])
def test_an_unreadable_board_ends_that_projects_turn_with_nothing_changed_and_the_others_go_on(
    root, owner, caplog, order,
):
    ops = _ops()
    _clean(OUTCOME, NEXT_ACTION, DECISION, REWRITTEN)
    garden, broken = _project(owner, "Garden Plan"), _project(owner, "Broken Plan")
    cards = {project["name"]: _card(root, project, "Choose the room") for project in (garden, broken)}
    _settle(root)
    for project in (garden, broken):
        _block(root, project, cards[project["name"]], "needs_input")
    _garble(_board_db(root, broken["board"]))
    _order(root, *((broken, garden) if order == "broken_first" else (garden, broken)))
    before = _state(root, broken["project_id"])
    asked = len(_tasks_of(root, broken, archived=True))

    with caplog.at_level(logging.WARNING, logger=ops.logger.name):
        assert ops.summary_tick(now=T0 + 100) == 1
    _assert_rewrite(root, garden, asked_at=T0 + 100)
    assert len(_tasks_of(root, broken, archived=True)) == asked
    assert _state(root, broken["project_id"]) == before
    warned = [record.getMessage() for record in caplog.records if record.levelno == logging.WARNING]
    assert [text for text in warned if broken["project_id"] in text]


def test_an_unreadable_ledger_ends_the_turn_with_nothing_changed_until_it_can_be_read(root, owner, caplog):
    ops = _ops()
    _clean(OUTCOME, NEXT_ACTION, DECISION, REWRITTEN)
    garden = _project(owner, "Garden Plan")
    pid = garden["project_id"]
    asking = _card(root, garden, "Choose the room")
    _settle(root)
    _Ledger().release("elsewhere", card="t_nobody", board="elsewhere")
    ledger = root / LEDGER_FILE
    assert not Path(str(ledger) + "-wal").exists()
    intact = ledger.read_bytes()
    _block(root, garden, asking, "needs_input")
    _garble(ledger)
    before = _state(root, pid)
    asked = len(_tasks_of(root, garden, archived=True))

    with caplog.at_level(logging.WARNING, logger=ops.logger.name):
        assert ops.summary_tick(now=T0 + 100) == 0
    assert _state(root, pid) == before
    assert len(_tasks_of(root, garden, archived=True)) == asked
    warned = [record.getMessage() for record in caplog.records if record.levelno == logging.WARNING]
    assert [text for text in warned if pid in text]

    ledger.write_bytes(intact)
    assert ops.summary_tick(now=T0 + 200) == 1
    _assert_rewrite(root, garden, asked_at=T0 + 200)
    assert _state(root, pid)["release_cursor"] == _ledger_top(root) > 0


# The tick runs inside a worker, and still reaches every store it reads: the boards, the root registry, the ledger.


def _count(path: Path, sql: str, params=()) -> int:
    return _rows(path, sql, params)[0][0]


def _still_to_read(root: Path, project: dict, state: dict) -> dict:
    """Rows past the Project's cursors in the three stores, counted with a plain connection and plain SQL."""
    board = _board_db(root, project["board"])
    return {
        "events": _count(board, "SELECT COUNT(*) FROM task_events WHERE id > ?", (state["event_cursor"],)),
        "merges": _count(
            board,
            "SELECT COUNT(*) FROM kanban_deliveries WHERE pull_request_state = 'merged' AND id > ?",
            (state["delivery_cursor"],),
        ),
        "releases": _count(
            root / LEDGER_FILE, "SELECT COUNT(*) FROM release_events WHERE id > ?", (state["release_cursor"],),
        ),
    }


def test_the_tick_under_the_worker_environment_reaches_every_store_it_reads(root, owner, monkeypatch):
    ops = _ops()
    _clean(OUTCOME, NEXT_ACTION, DECISION, REWRITTEN)
    garden = _project(owner, "Garden Plan")
    desk = _project(owner, "Planning Desk")
    spare = _project(owner, "Spare Room")
    projects = (garden, desk, spare)
    asking = _card(root, garden, "Choose the room")
    lamp = _card(root, desk, "Order the lamp")
    worker_task = _card(root, desk, "Plan the next milestone", assignee=PLANNER)
    parcel = _card(root, spare, "Ship the parcel")
    _settle(root)

    _block(root, garden, asking, "needs_input")
    _complete(root, desk, lamp)
    _Ledger().release("lamp", card=lamp, board=desk["board"])
    merged = _delivery(root, spare, parcel, "merged")
    profile_home = root / "profiles" / PLANNER
    profile_home.mkdir(parents=True)
    with projects_db.connect_closing(profile_home / REGISTRY_FILE):
        pass

    before = {project["name"]: _state(root, project["project_id"]) for project in projects}
    waiting = {project["name"]: _still_to_read(root, project, before[project["name"]]) for project in projects}
    assert waiting["Garden Plan"]["events"] > 0
    assert waiting["Planning Desk"]["events"] > 0 and waiting["Planning Desk"]["releases"] > 0
    assert waiting["Spare Room"]["merges"] == 1
    held = {
        project["name"]: (
            _count(_board_db(root, project["board"]), "SELECT COUNT(*) FROM task_events"),
            _count(_board_db(root, project["board"]), "SELECT COUNT(*) FROM kanban_deliveries"),
        )
        for project in projects
    }
    ledger_rows = _count(root / LEDGER_FILE, "SELECT COUNT(*) FROM release_events")

    _enter_planning_worker(monkeypatch, root, desk["board"], worker_task)
    pinned = {name: os.environ[name] for name in ("HERMES_KANBAN_DB", "HERMES_KANBAN_BOARD", "HERMES_KANBAN_TASK")}
    assert Path(pinned["HERMES_KANBAN_DB"]).resolve() == _board_db(root, desk["board"]).resolve()

    assert ops.summary_tick(now=T0 + 500) == 3

    assert sorted(task.title for task in _summary_tasks(root)) == sorted(TITLE + p["name"] for p in projects)
    top_of_ledger = _count(root / LEDGER_FILE, "SELECT MAX(id) FROM release_events")
    for project in projects:
        state = _state(root, project["project_id"])
        board = _board_db(root, project["board"])
        assert state["event_cursor"] == _count(board, "SELECT MAX(id) FROM task_events")
        assert state["delivery_cursor"] == _count(
            board, "SELECT COALESCE(MAX(id), 0) FROM kanban_deliveries WHERE pull_request_state = 'merged'",
        )
        assert state["release_cursor"] == top_of_ledger
        assert _still_to_read(root, project, state) == {"events": 0, "merges": 0, "releases": 0}
        assert (state["pending"], state["pending_urgent"], state["failures"]) == (0, 0, 0)
        assert state["open_task_id"] is not None
        assert (
            _count(board, "SELECT COUNT(*) FROM task_events"), _count(board, "SELECT COUNT(*) FROM kanban_deliveries"),
        ) == held[project["name"]]
    assert _state(root, spare["project_id"])["delivery_cursor"] == merged
    assert _count(root / LEDGER_FILE, "SELECT COUNT(*) FROM release_events") == ledger_rows
    assert _count(profile_home / REGISTRY_FILE, "SELECT COUNT(*) FROM project_summary_state") == 0
    assert {name: os.environ[name] for name in pinned} == pinned


# One table sorts the kinds of events, and nothing else decides what an event causes.


def test_one_table_sorts_every_kind_of_event():
    events = _events()
    material, urgent, ignored = events.MATERIAL, events.URGENT, events.IGNORED
    assert len({material, urgent, ignored}) == 3
    assert isinstance(events.EVENT_CLASS, dict)
    listed = {
        "created": material, "linked": material, "unlinked": material, "decomposed": material,
        "archived": material, "completed": material, "blocked": material, "unblocked": material,
        "review_requested": urgent, "merged": urgent, "released": urgent,
        "claimed": ignored, "spawned": ignored, "heartbeat": ignored, "commented": ignored, "promoted": ignored,
    }
    for kind, expected in listed.items():
        assert events.classify(kind) == expected, kind
    assert events.classify("blocked", "needs_input") == urgent
    assert events.classify("unblocked", "needs_input") == urgent
    for detail in ("capability", "dependency", "transient", None):
        assert events.classify("blocked", detail) == material, detail
        assert events.classify("unblocked", detail) == material, detail
    for kind in ("delivery_merged", "dependency_wait", "block_loop_detected", "owner_retry", "no_such_kind", ""):
        assert events.classify(kind) == ignored, kind


def test_the_table_alone_decides_what_a_progress_event_causes(root, owner, monkeypatch):
    ops, events = _ops(), _events()
    _clean(OUTCOME, NEXT_ACTION, DECISION, REWRITTEN)
    garden = _project(owner, "Garden Plan")
    busy, plain, fresh = (_card(root, garden, title) for title in ("Busy card", "Plain card", "Fresh card"))
    _settle(root)

    _quiet(root, garden, busy)
    assert ops.summary_tick(now=T0 + 100) == 0
    assert _flags(root, garden) == (0, 0)
    monkeypatch.setitem(events.EVENT_CLASS, "heartbeat", events.URGENT)
    _heartbeat(root, garden, busy)
    assert ops.summary_tick(now=T0 + 200) == 1
    stored = _assert_rewrite(root, garden, asked_at=T0 + 200)
    _store_rewrite(root, stored, T0 + 210)

    monkeypatch.setitem(events.EVENT_CLASS, "completed", events.IGNORED)
    _complete(root, garden, plain)
    assert ops.summary_tick(now=T0 + 200 + 3 * HOUR) == 0
    assert _flags(root, garden) == (0, 0)
    assert _tasks_of(root, garden) == []
    _archive_card(root, garden, fresh)
    assert ops.summary_tick(now=T0 + 200 + 4 * HOUR) == 1
    _assert_rewrite(root, garden, asked_at=T0 + 200 + 4 * HOUR)


@pytest.mark.parametrize("source", ["merged", "released"])
def test_a_merge_and_a_release_are_sorted_by_the_same_table(root, owner, monkeypatch, source):
    ops, events = _ops(), _events()
    _clean(OUTCOME, NEXT_ACTION, DECISION, REWRITTEN)
    garden = _project(owner, "Garden Plan")
    pid = garden["project_id"]
    card, asking = _card(root, garden, "Plant the beds"), _card(root, garden, "Choose the room")
    _settle(root)
    assert events.EVENT_CLASS[source] == events.URGENT

    monkeypatch.setitem(events.EVENT_CLASS, source, events.IGNORED)
    if source == "merged":
        merge = _delivery(root, garden, card, "merged")
    else:
        _Ledger().release("beds", card=card, board=garden["board"])
    assert ops.summary_tick(now=T0 + 100) == 0
    assert _flags(root, garden) == (0, 0)
    assert _tasks_of(root, garden) == []
    state = _state(root, pid)
    assert state["delivery_cursor" if source == "merged" else "release_cursor"] > 0

    _block(root, garden, asking, "needs_input")
    assert ops.summary_tick(now=T0 + 200) == 1
    _assert_rewrite(root, garden, asked_at=T0 + 200)
    if source == "merged":
        assert _state(root, pid)["delivery_cursor"] == merge


# The module on its own: its sources may be missing, and a swap moves the cursors only from the values it read.


def test_a_missing_deliveries_table_and_a_missing_ledger_are_sources_without_events(root, owner):
    events = _events()
    garden = _project(owner, "Garden Plan")
    for title in ("Plant the beds", "Book the room"):
        _card(root, garden, title)
    _block(root, garden, _card(root, garden, "Choose the room"), "needs_input")
    board = _board_db(root, garden["board"])
    with contextlib.closing(sqlite3.connect(board)) as conn:
        conn.execute("DROP TABLE kanban_deliveries")
    assert not (root / LEDGER_FILE).exists()

    batch = events.read_events(_listed(root, garden), events.Cursors(0, 0, 0))
    assert batch.cursors == events.Cursors(_top(root, garden), 0, 0)
    assert batch.urgent == 1
    assert batch.material == _count(board, "SELECT COUNT(*) FROM task_events WHERE kind = 'created'") >= 3
    assert not (root / LEDGER_FILE).exists()
    again = events.read_events(_listed(root, garden), batch.cursors)
    assert again == events.Batch(batch.cursors, 0, 0)


def test_a_project_without_a_board_or_on_the_summary_board_has_no_events(root, owner):
    events = _events()
    _clean(OUTCOME, NEXT_ACTION, DECISION)
    garden = _project(owner, "Garden Plan")
    _card(root, garden)
    _settle(root)
    assert _top(root, {"board": BOARD}) > 0
    nothing = events.Batch(events.Cursors(0, 0, 0), 0, 0)
    for name, columns in (("Loop", {"board_slug": BOARD}), ("Bare", {})):
        project = {"project_id": _registry_project(root, name, **columns)}
        assert events.read_events(_listed(root, project), events.Cursors(0, 0, 0)) == nothing, name


def test_the_swap_moves_the_cursors_once_and_only_from_the_values_it_read(root, owner):
    events = _events()
    _clean(OUTCOME, NEXT_ACTION, DECISION)
    garden = _project(owner, "Garden Plan")
    pid = garden["project_id"]
    _card(root, garden)
    _settle(root)
    stored = _state(root, pid)
    old = events.Cursors(stored["event_cursor"], stored["delivery_cursor"], stored["release_cursor"])
    won = events.Batch(events.Cursors(old.event + 5, old.delivery + 1, old.release + 2), material=1, urgent=0)
    lost = events.Batch(events.Cursors(old.event + 9, old.delivery, old.release), material=0, urgent=1)

    with projects_db.connect_closing(root / REGISTRY_FILE) as first, projects_db.connect_closing(
        root / REGISTRY_FILE
    ) as second:
        assert events.advance_cursors(first, pid, old, won) is True
        moved = _state(root, pid)
        assert events.Cursors(moved["event_cursor"], moved["delivery_cursor"], moved["release_cursor"]) == won.cursors
        assert (moved["pending"], moved["pending_urgent"], moved["updated_at"]) == (1, 0, stored["updated_at"])

        assert events.advance_cursors(second, pid, old, lost) is False
        assert _state(root, pid) == moved

        quiet = events.Batch(events.Cursors(old.event + 6, old.delivery + 1, old.release + 2), material=0, urgent=0)
        assert events.advance_cursors(second, pid, won.cursors, quiet) is True
        later = _state(root, pid)
        assert (later["event_cursor"], later["pending"], later["pending_urgent"]) == (old.event + 6, 1, 0)

        with kanban_db.write_txn(second):
            second.execute("UPDATE project_summary_state SET open_task_id = 't_open' WHERE project_id = ?", (pid,))
        held = _state(root, pid)
        assert events.advance_cursors(first, pid, quiet.cursors, lost) is False
        assert _state(root, pid) == held
