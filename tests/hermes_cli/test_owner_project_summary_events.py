"""Card B of the Project summary writer plan: Raphael refreshes a Project's summary after material events.

The summary tick reads three sources through the cursors of the Project's state row: the task events of the
Project's own board, its merged delivery rows and the release ledger. One table sorts every event into material,
urgent or ignored. Material events are rewritten at most once an hour, a decision, a merge and a release at once,
and ten quick changes are one rewrite (owner decision of 2026-10-27). Card A's owner rules 1 to 3 hold for every
rewrite request: the kernel's time box, one open task found by one check first, and a failure limit that only delays
the next request by a day.

The tests reach the new module only through ``summary_tick``, which each one imports for itself; the tests of the
covered line (T2 and T3, and the last ones, on a finished task handed back for its request key) also read it with the
module's own parser. The last ones also run once under the dispatcher's worker environment.
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
from types import SimpleNamespace

import pytest

from hermes_cli import kanban_db, kanban_risk_tier, owner_workspace as ow, projects_db, release_ledger
from hermes_cli.owner_text import check_owner_text
from plugins.dashboard_auth.raphael_workspace import model_policy

PLANNER = "raphael-planner"
BOARD = "project-summaries"
TITLE = "Write the Project summary: "
HOUR = 3600
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


def _complete_the_removal(root: Path, project_id: str) -> None:
    """A permanent removal that has completed: the registry keeps the Project's row, with the phase done."""
    with projects_db.connect_closing(root / REGISTRY_FILE) as registry:
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

    # The hour runs from the last request, T0 + HOUR, and the idle tick above did not move it: a change is due at once.
    _card(root, garden, "One more step")
    assert ops.summary_tick(now=T0 + 9 * HOUR + 60) == 1
    _assert_rewrite(root, garden, asked_at=T0 + 9 * HOUR + 60)


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
    """An event the kernel writes next to other work, here on its own: ``delivery_merged_on_base`` and the like."""
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
    for kind in ("promoted", "delivery_merged_on_base", "dependency_wait", "block_loop_detected", "owner_retry"):
        _append(root, garden, busy, kind, {"note": "written next to other work"})
    made = set(_kinds(root, garden, after=before))
    assert {"claimed", "spawned", "heartbeat", "commented", "promoted", "delivery_merged_on_base"} <= made
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


# Test 10: a Project that is gone is skipped before any read, with no failure; a live Project in the same tick is asked.


@pytest.mark.parametrize("order", ["gone_first", "live_first"])
def test_a_project_that_is_gone_is_skipped_without_a_read_or_a_failure_while_a_live_one_is_asked(
    root, owner, monkeypatch, caplog, order,
):
    ops = _ops()
    _clean(OUTCOME, NEXT_ACTION, DECISION, REWRITTEN)
    live, gone = _project(owner, "Garden Plan"), _project(owner, "Old Plan")
    cards = {project["name"]: _card(root, project, "Choose the room") for project in (live, gone)}
    _settle(root)
    for project in (live, gone):
        _block(root, project, cards[project["name"]], "needs_input")
    _complete_the_removal(root, gone["project_id"])
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


# Rework after the review of the first hand-over (2026-10-28): two major findings, one test (T1 to T3) for each case.


class _ProcessDied(BaseException):
    """A gateway restart as the interrupted code sees it: nothing after this point runs."""


def _merge_delivery(root: Path, project: dict, delivery: int, card: str, ledger: _Ledger) -> None:
    """A merge as kanban_delivery._merged leaves it: the change in the release ledger, then the row's state and the
    receipt ``delivery_merged`` on its source card, written in one transaction."""
    label = f"merge-{delivery}"
    ledger.merge(label, card=card, board=project["board"])
    with _board(root, project["board"]) as conn, kanban_db.write_txn(conn):
        conn.execute("UPDATE kanban_deliveries SET pull_request_state = 'merged' WHERE id = ?", (delivery,))
        kanban_db._append_event(
            conn, card, "delivery_merged",
            {
                "delivery_id": delivery, "head": _sha(label + "-head"), "pull_request_number": delivery,
                "repository": "mtbitcr/hermes-agent", "merge_commit": _sha(label), "recorded": True,
                "attempt": _sha(label + "-attempt"),
            },
        )


# T1: delivery rows are updated in place, so merges need not finish in row-id order; every merge is rewritten at once.


def test_merges_in_the_reverse_order_of_their_rows_each_give_a_task_at_once(root, owner):
    ops = _ops()
    _clean(OUTCOME, NEXT_ACTION, DECISION, REWRITTEN)
    garden = _project(owner, "Garden Plan")
    pid = garden["project_id"]
    cards = [_card(root, garden, "Plant the beds"), _card(root, garden, "Water the beds")]
    for card in cards:
        _complete(root, garden, card)
    older, newer = (_delivery(root, garden, card, "open") for card in cards)
    assert older < newer
    _settle(root)
    assert _state(root, pid)["delivery_cursor"] == 0
    ledger = _Ledger()

    _merge_delivery(root, garden, newer, cards[1], ledger)
    assert ops.summary_tick(now=T0 + 100) == 1
    first = _assert_rewrite(root, garden, asked_at=T0 + 100)
    assert _state(root, pid)["delivery_cursor"] == newer
    _store_rewrite(root, first, T0 + 110)
    assert _summary_of(root, pid)["revision"] == 2
    assert _tasks_of(root, garden) == []

    _merge_delivery(root, garden, older, cards[0], ledger)
    assert ops.summary_tick(now=T0 + 200) == 1
    second = _assert_rewrite(root, garden, asked_at=T0 + 200)
    assert second != first
    _store_rewrite(root, second, T0 + 210)
    assert _summary_of(root, pid)["revision"] == 3
    assert _tasks_of(root, garden) == []

    assert ops.summary_tick(now=T0 + 300) == 0
    assert _tasks_of(root, garden) == []
    assert _flags(root, garden) == (0, 0)


def _cursors(state: dict) -> tuple:
    return (state["event_cursor"], state["delivery_cursor"], state["release_cursor"])


# T2: a task is the receipt of what it covers. A publisher whose task another tick recorded first still moves the
# cursors, so the events the task covers are not counted again once it is harvested.


def test_a_publisher_whose_task_another_tick_recorded_first_still_moves_the_cursors(root, owner, monkeypatch):
    from hermes_cli import owner_project_summary_events as events

    ops = _ops()
    _clean(OUTCOME, NEXT_ACTION, DECISION, REWRITTEN)
    garden = _project(owner, "Garden Plan")
    pid = garden["project_id"]
    asking = _card(root, garden, "Choose the room")
    _settle(root)
    settled = _state(root, pid)
    _block(root, garden, asking, "needs_input")
    top = _top(root, garden)
    assert top > settled["event_cursor"]

    create_task = kanban_db.create_task
    met: list = []

    def create_then_let_another_tick_in(*args, **kwargs):
        """Tick A's task is published and committed; before A's claim a second tick meets it on its own connections."""
        task_id = create_task(*args, **kwargs)
        if not met:
            met.append(task_id)
            assert ops.summary_tick(now=T0 + 100) == 0
        return task_id

    monkeypatch.setattr(kanban_db, "create_task", create_then_let_another_tick_in)
    assert ops.summary_tick(now=T0 + 100) == 0
    monkeypatch.setattr(kanban_db, "create_task", create_task)

    [task] = _tasks_of(root, garden)
    assert (task.id, task.status) == (met[0], "ready")
    assert _open_task_ids(root, garden) == [task.id]
    adopted = _state(root, pid)
    assert (adopted["open_task_id"], adopted["last_requested_at"], adopted["failures"]) == (task.id, T0 + 100, 0)
    assert (adopted["pending"], adopted["pending_urgent"]) == (0, 0)
    assert _cursors(adopted) == (top, 0, 0)
    assert events.covered_cursors(task.body) == events.Cursors(top, 0, 0)

    _finish(root, task.id, _valid(outcome=REWRITTEN))
    assert ops.summary_tick(now=T0 + 110) == 0
    assert _summary_of(root, pid)["revision"] == 2
    assert _tasks_of(root, garden) == []
    assert _flags(root, garden) == (0, 0)
    assert _cursors(_state(root, pid)) == (top, 0, 0)

    _unblock(root, garden, asking)
    assert ops.summary_tick(now=T0 + 120) == 1
    _assert_rewrite(root, garden, asked_at=T0 + 120)
    assert _cursors(_state(root, pid)) == (_top(root, garden), 0, 0)


# T3: the same when the publisher stops after it published. The task names what it covers, so a later tick adopts it
# through the one check, and the harvest of the task moves the cursors.


def test_a_task_published_by_a_tick_that_stopped_covers_its_events_once_a_later_tick_adopts_it(
    root, owner, monkeypatch,
):
    from hermes_cli import owner_project_summary_events as events

    ops = _ops()
    _clean(OUTCOME, NEXT_ACTION, DECISION, REWRITTEN)
    garden = _project(owner, "Garden Plan")
    pid = garden["project_id"]
    asking = _card(root, garden, "Choose the room")
    _settle(root)
    settled = _state(root, pid)
    _block(root, garden, asking, "needs_input")
    top = _top(root, garden)
    assert top > settled["event_cursor"]

    create_task = kanban_db.create_task

    def create_then_die(*args, **kwargs):
        create_task(*args, **kwargs)
        raise _ProcessDied

    monkeypatch.setattr(kanban_db, "create_task", create_then_die)
    with pytest.raises(_ProcessDied):
        ops.summary_tick(now=T0 + 100)
    monkeypatch.setattr(kanban_db, "create_task", create_task)
    assert _state(root, pid) == settled
    [task] = _tasks_of(root, garden)
    assert task.status == "ready"

    assert ops.summary_tick(now=T0 + 150) == 0
    adopted = _state(root, pid)
    assert (adopted["open_task_id"], adopted["last_requested_at"], adopted["failures"]) == (task.id, T0 + 150, 0)
    assert (adopted["pending"], adopted["pending_urgent"]) == (0, 0)
    assert _open_task_ids(root, garden) == [task.id]

    _finish(root, task.id, _valid(outcome=REWRITTEN))
    assert ops.summary_tick(now=T0 + 160) == 0
    assert _summary_of(root, pid)["revision"] == 2
    assert _tasks_of(root, garden) == []
    assert _flags(root, garden) == (0, 0)
    assert _cursors(_state(root, pid)) == (top, 0, 0)
    assert events.covered_cursors(task.body) == events.Cursors(top, 0, 0)

    _unblock(root, garden, asking)
    assert ops.summary_tick(now=T0 + 170) == 1
    _assert_rewrite(root, garden, asked_at=T0 + 170)
    assert _cursors(_state(root, pid)) == (_top(root, garden), 0, 0)


# Rework after the review of the second hand-over (2026-10-28), owner rule 2: a task clears only the pending work of
# the events within its own watermark. A delayed tick can store the events of a task that was published and never
# recorded; idempotent creation then hands that finished task back to a request whose cursors are already past what
# the task covers. The events after the watermark must stay pending, and the cursors must never go back.


def _tops(root: Path, project: dict) -> tuple:
    """Where the three cursors stand once everything is read, counted with plain connections."""
    merged = _rows(
        _board_db(root, project["board"]),
        "SELECT COALESCE(MAX(id), 0) FROM kanban_deliveries WHERE pull_request_state = 'merged'",
    )[0][0]
    return (_top(root, project), merged, _ledger_top(root) if (root / LEDGER_FILE).exists() else 0)


_URGENT_KINDS = ("decision", "merge", "release")


def _scene(root: Path, owner, *, failed: bool, others: tuple = (), worker_on: str | None = None) -> SimpleNamespace:
    """A settled Project with a card for each kind of change, and the time its next request falls due.

    With ``failed`` its first rewrite has failed, and the request is due once the backoff of an hour has passed;
    otherwise it is due once the hour since its last request has. ``others`` are settled Projects that stay quiet.
    ``worker_on`` names one of them, on whose board a worker's own card is made before the settle.
    """
    ops = _ops()
    _clean(OUTCOME, NEXT_ACTION, DECISION, REWRITTEN)
    garden = _project(owner, "Garden Plan")
    ledger = _Ledger()
    cards = {
        name: _card(root, garden, name)
        for name in ("opening", "finish", "finish again", "decision", "merge", "release")
    }
    delivery = _delivery(root, garden, cards["merge"], "open")
    beside = {name: _project(owner, name) for name in others}
    quiet = {name: _card(root, project, "A quiet card") for name, project in beside.items()}
    worker_task = None
    if worker_on is not None:
        with _board(root, beside[worker_on]["board"]) as conn:
            worker_task = kanban_db.create_task(
                conn, title="Plan the next milestone", assignee=PLANNER, project_id=beside[worker_on]["project_id"],
            )
    _settle(root)

    due = T0 + HOUR
    if failed:
        _complete(root, garden, cards["opening"])
        assert ops.summary_tick(now=T0 + HOUR) == 1
        _fail(root, _assert_rewrite(root, garden, asked_at=T0 + HOUR))
        assert ops.summary_tick(now=T0 + HOUR + 10) == 0
        failure = _state(root, garden["project_id"])
        assert (failure["failures"], failure["pending"], failure["open_task_id"]) == (1, 1, None)
        due = failure["updated_at"] + HOUR
    happen = {
        "finish": lambda: _complete(root, garden, cards["finish"]),
        "finish again": lambda: _complete(root, garden, cards["finish again"]),
        "decision": lambda: _block(root, garden, cards["decision"], "needs_input"),
        "merge": lambda: _merge_delivery(root, garden, delivery, cards["merge"], ledger),
        "release": lambda: ledger.release("lamp", card=cards["release"], board=garden["board"]),
    }
    return SimpleNamespace(
        garden=garden, pid=garden["project_id"], due=due, happen=happen, cards=cards,
        beside=beside, quiet=quiet, worker_task=worker_task,
    )


def _race(root: Path, scene: SimpleNamespace, monkeypatch, later) -> kanban_db.Task:
    """The delayed tick of the review; returns the finished task that nobody has recorded.

    Tick B passes the one check and waits before it reads events. Meanwhile tick A, at the due time, publishes its task
    and stops before it records it, the worker finishes that task, and ``later`` happens. B then goes on with its own
    clock, a second before the due time: it is not due, so it stores the events it reads and moves the cursors.
    """
    from hermes_cli import owner_project_summary_events as events

    ops = _ops()
    read, create_task = events.read, kanban_db.create_task
    published: list = []
    watermark: list = []

    def create_then_die(*args, **kwargs):
        published.append(create_task(*args, **kwargs))
        raise _ProcessDied

    def read_late(registry, project):
        if project.id == scene.pid and not watermark:
            watermark.append(_tops(root, scene.garden))
            monkeypatch.setattr(kanban_db, "create_task", create_then_die)
            with pytest.raises(_ProcessDied):
                ops.summary_tick(now=scene.due)
            monkeypatch.setattr(kanban_db, "create_task", create_task)
            _finish(root, published[0], _valid(outcome=REWRITTEN))
            later()
        return read(registry, project)

    monkeypatch.setattr(events, "read", read_late)
    assert ops.summary_tick(now=scene.due - 1) == 0
    monkeypatch.setattr(events, "read", read)

    [task] = _tasks_of(root, scene.garden)
    assert (task.id, task.status) == (published[0], "done")
    recorded = _state(root, scene.pid)
    assert (recorded["open_task_id"], recorded["pending"]) == (None, 1)
    assert events.covered_cursors(task.body) == events.Cursors(*watermark[0])
    assert _cursors(recorded) == _tops(root, scene.garden)
    assert _cursors(recorded) != watermark[0]
    return task


def _reuse_and_harvest(root: Path, scene: SimpleNamespace, monkeypatch, task, *, urgent: bool) -> None:
    """The next request gets the finished task back for its key, and the harvest follows.

    The task covers only what lies up to its watermark, so the events after it stay pending: an urgent one is asked for
    at once after the harvest, an ordinary one once the hour since the claim has passed, and each exactly once.
    """
    ops = _ops()
    garden, pid = scene.garden, scene.pid
    recorded = _state(root, pid)
    asked = len(_tasks_of(root, garden, archived=True))
    claim_at = scene.due + 5

    with _Trace(monkeypatch, root) as trace:
        assert ops.summary_tick(now=claim_at) == 1
    _assert_the_check_comes_first(trace)
    claimed = _state(root, pid)
    assert claimed["open_task_id"] == task.id
    assert (claimed["last_requested_at"], claimed["failures"]) == (claim_at, recorded["failures"])
    assert _cursors(claimed) == _cursors(recorded)
    assert (claimed["pending"], claimed["pending_urgent"]) == (1, int(urgent))
    assert len(_tasks_of(root, garden, archived=True)) == asked

    harvest_at = claim_at + 10
    if urgent:
        assert ops.summary_tick(now=harvest_at) == 1
        asked_at = harvest_at
    else:
        assert ops.summary_tick(now=harvest_at) == 0
        harvested = _state(root, pid)
        assert (harvested["open_task_id"], harvested["failures"]) == (None, 0)
        assert (harvested["pending"], harvested["pending_urgent"]) == (1, 0)
        assert ops.summary_tick(now=claim_at + HOUR - 1) == 0
        assert len(_tasks_of(root, garden, archived=True)) == asked
        asked_at = claim_at + HOUR
        assert ops.summary_tick(now=asked_at) == 1
    assert _summary_of(root, pid)["revision"] == 2
    second = _assert_rewrite(root, garden, asked_at=asked_at)
    assert second != task.id
    assert len(_tasks_of(root, garden, archived=True)) == asked + 1
    assert _cursors(_state(root, pid)) == _tops(root, garden)

    _store_rewrite(root, second, asked_at + 10)
    assert _summary_of(root, pid)["revision"] == 3
    for at in (asked_at + 20, asked_at + 2 * HOUR):
        assert ops.summary_tick(now=at) == 0
    assert _tasks_of(root, garden) == []
    assert len(_tasks_of(root, garden, archived=True)) == asked + 1
    assert _flags(root, garden) == (0, 0)
    assert _cursors(_state(root, pid)) == _tops(root, garden)


# What the task covers is a decision (material case), or a finish; what comes after it is the case's own event. The
# failure is counted for the cases that cross the backoff, and not for the hourly boundary.
_REUSE_CASES = {
    "material": ("decision", "finish", True),
    "decision": ("finish", "decision", True),
    "merge": ("finish", "merge", True),
    "release": ("finish", "release", True),
    "hour": ("finish", "finish again", False),
}


@pytest.mark.parametrize("case", list(_REUSE_CASES))
def test_a_finished_task_handed_back_for_its_key_clears_only_the_events_it_covers(root, owner, monkeypatch, case):
    covered, later, failed = _REUSE_CASES[case]
    scene = _scene(root, owner, failed=failed)
    scene.happen[covered]()
    task = _race(root, scene, monkeypatch, scene.happen[later])
    _reuse_and_harvest(root, scene, monkeypatch, task, urgent=later in _URGENT_KINDS)


def test_the_flags_are_kept_when_the_events_after_the_tasks_line_cannot_be_read(root, owner, monkeypatch):
    scene = _scene(root, owner, failed=True)
    scene.happen["finish"]()
    task = _race(root, scene, monkeypatch, scene.happen["merge"])
    before = _state(root, scene.pid)
    _garble(root / LEDGER_FILE)

    assert _ops().summary_tick(now=scene.due + 5) == 1
    claimed = _state(root, scene.pid)
    assert (claimed["open_task_id"], claimed["pending"], claimed["pending_urgent"]) == (task.id, 1, 1)
    assert _cursors(claimed) == _cursors(before)


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


def test_a_finished_task_handed_back_for_its_key_holds_under_the_worker_environment(root, owner, monkeypatch):
    scene = _scene(root, owner, failed=True, others=("Planning Desk", "Spare Room"), worker_on="Planning Desk")
    desk, spare = scene.beside["Planning Desk"], scene.beside["Spare Room"]
    scene.happen["finish"]()
    for name, project in scene.beside.items():
        _quiet(root, project, scene.quiet[name])
        assert _cursors(_state(root, project["project_id"])) != _tops(root, project)

    profile_home = root / "profiles" / PLANNER
    profile_home.mkdir(parents=True)
    with projects_db.connect_closing(profile_home / REGISTRY_FILE):
        pass
    _enter_planning_worker(monkeypatch, root, desk["board"], scene.worker_task)
    pinned = {name: os.environ[name] for name in ("HERMES_KANBAN_DB", "HERMES_KANBAN_BOARD")}
    own_board = _board_db(root, desk["board"])
    assert Path(pinned["HERMES_KANBAN_DB"]).resolve() == own_board.resolve()
    own_cards = _rows(own_board, "SELECT id, title, status FROM tasks ORDER BY id")

    task = _race(root, scene, monkeypatch, scene.happen["merge"])
    _reuse_and_harvest(root, scene, monkeypatch, task, urgent=True)

    registry = root / REGISTRY_FILE
    projects = (scene.garden, desk, spare)
    assert _rows(
        registry,
        "SELECT project_id, event_cursor, delivery_cursor, release_cursor, pending, pending_urgent, open_task_id, "
        "failures FROM project_summary_state ORDER BY project_id",
    ) == sorted((project["project_id"], *_tops(root, project), 0, 0, None, 0) for project in projects)
    assert _rows(registry, "SELECT project_id, revision FROM project_summaries ORDER BY project_id") == sorted(
        [(scene.pid, 3), (desk["project_id"], 1), (spare["project_id"], 1)]
    )
    assert _rows(
        _board_db(root, BOARD),
        "SELECT title, count(*) FROM tasks WHERE task_kind = 'work' GROUP BY title ORDER BY title",
    ) == [(TITLE + "Garden Plan", 4), (TITLE + "Planning Desk", 1), (TITLE + "Spare Room", 1)]
    assert _rows(own_board, "SELECT id, title, status FROM tasks ORDER BY id") == own_cards
    assert _rows(profile_home / REGISTRY_FILE, "SELECT count(*) FROM project_summaries") == [(0,)]
    assert _rows(profile_home / REGISTRY_FILE, "SELECT count(*) FROM project_summary_state") == [(0,)]
    assert {name: os.environ[name] for name in pinned} == pinned
