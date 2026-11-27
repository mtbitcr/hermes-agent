"""Work that names files the owner must give waits until the owner gave them.

Card C2 of the named inputs plan. ``claim_task`` makes one call into
``hermes_cli.owner_named_inputs`` after its parents check: a ready card whose
declared inputs the owner has not all attached is blocked ``needs_input`` with
the fixed hold sentence in the claim's own transaction, no run starts, the
claim returns None, and ``kanban_task_blocked`` fires after the commit the way
``block_task`` fires it. A card without a declaration is claimed as today.

Only a Decisions reader that names ``owner_input_upload_v1`` learns what is
missing: each owner-input item of a declaring card then carries
``missing_inputs``, each still-missing input's number, name, plain type label
and the size limit, never a file name. Every other reader gets today's keys.
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
from types import SimpleNamespace

import pytest
from aiohttp.test_utils import TestClient, TestServer

from hermes_cli import kanban_db as kb
from hermes_cli import owner_named_inputs as oni
from hermes_cli import owner_workspace as ow
from hermes_cli.owner_text import check_owner_text
from hermes_cli.plugins import get_plugin_manager
from hermes_cli.profiles import get_active_profile_name

# The real authority and apply path of card C1 and the harnesses under it
# (temporary home, auto-approver, configured provider, the owner the gateway
# routes resolve) are reused verbatim, so every card here is a real one.
from tests.gateway.test_api_server_owner_waiting import (  # noqa: F401
    _app,
    _get,
    owner,
)
from tests.hermes_cli.test_owner_named_inputs import (
    _apply,
    _authorize,
    _observe,
    _plan_change,
    _store_file,
)
from tests.hermes_cli.test_owner_workspace import (  # noqa: F401
    _bootstrap_board,
    _configured_provider,
    ctx,
)

_PRICE_LIST = {"name": "price list", "type": "csv"}
_OPENING_HOURS = {"name": "opening hours", "type": "txt"}
_MENU = {"name": "menu", "type": "md"}
# The keys of every Decisions item today.
_DECISION_KEYS = {
    "decision_ref", "authority", "kind", "project_slug", "project_name",
    "title", "reason", "created_at",
}
_DECISIONS = "/v1/owner-workspace/decisions"


def _seen_on_named_board(board, task_id: str):
    """(status, number of 'blocked' events) of the card in the store of *board*.

    Read through a second, read-only connection. None when no board is named
    (never the current one), the store cannot be read or it holds no such card.
    It never raises: a hook that raises is swallowed and hides the failure.
    """
    if not board:
        return None
    try:
        store = str(kb.kanban_db_path(board=board))
        if not _observe(store, "SELECT COUNT(*) FROM tasks WHERE id = ?", task_id):
            return None
        return (
            _observe(store, "SELECT status FROM tasks WHERE id = ?", task_id),
            _observe(
                store,
                "SELECT COUNT(*) FROM task_events WHERE task_id = ? AND kind = 'blocked'",
                task_id,
            ),
        )
    except (OSError, ValueError, sqlite3.Error):
        return None


@pytest.fixture
def fired_hooks():
    """Each kanban_task_blocked and kanban_task_claimed call, in order.

    The registry ``invoke_hook`` reads is patched and restored, as
    tests/hermes_cli/test_kanban_lifecycle_hooks.py does. Each entry also holds
    the card's status as a second, read-only connection to ``store`` (set by
    the test) saw it while the hook ran: a hook fired inside the transaction
    would still see the card as it was before. ``on_named_board`` lists, call
    for call, what :func:`_seen_on_named_board` read on the board the call names.
    """
    manager = get_plugin_manager()
    saved = {name: list(callbacks) for name, callbacks in manager._hooks.items()}
    seen = SimpleNamespace(store=None, fired=[], on_named_board=[])

    def capture(hook: str, fields: dict) -> None:
        status = _observe(
            seen.store, "SELECT status FROM tasks WHERE id = ?", fields["task_id"],
        )
        seen.on_named_board.append(
            _seen_on_named_board(fields.get("board"), fields["task_id"])
        )
        # The manager adds its telemetry marker to every hook; keep only what
        # the caller passed.
        fields.pop("telemetry_schema_version", None)
        seen.fired.append((hook, fields, status))

    for hook in ("kanban_task_blocked", "kanban_task_claimed"):
        manager._hooks.setdefault(hook, []).append(
            lambda _hook=hook, **fields: capture(_hook, fields)
        )
    try:
        yield seen
    finally:
        manager._hooks = saved


def _card(ctx, *inputs: dict) -> tuple[dict, str]:
    """(Project, card) of one approved plan change that adds one ready card.

    The card declares *inputs* through the real authority and apply path;
    without any it declares nothing.
    """
    setup = _bootstrap_board(ctx)
    extra = {"inputs": list(inputs)} if inputs else {}
    result = _apply(ctx, _authorize(
        *_plan_change(setup["project_id"], **extra),
        project_id=setup["project_id"],
    ))
    assert result["ok"] is True
    (task_id,) = result["created_task_ids"]
    return setup, task_id


def _late_card(ctx, *inputs: dict) -> tuple[dict, str]:
    """(Project, card) of :func:`_card` whose *inputs* are declared after its activation.

    The hold at the activation never sees such a declaration, so the card is
    ready and only the hold at the claim holds it.
    """
    setup, task_id = _card(ctx)
    with contextlib.closing(kb.connect(board=setup["board"])) as conn:
        with kb.write_txn(conn):
            kb._append_event(conn, task_id, oni.DECLARED_EVENT, oni._declaration(list(inputs)))
    return setup, task_id


def _give(conn, board: str, task_id: str, filename: str, uploaded_by=oni.OWNER_UPLOADER):
    """One file attached to the card by *uploaded_by*."""
    kb.store_attachment_bytes(
        conn, task_id, filename, b"data\n", uploaded_by=uploaded_by, board=board,
    )


def _status(conn, task_id: str) -> str:
    return conn.execute("SELECT status FROM tasks WHERE id = ?", (task_id,)).fetchone()[0]


def _events(conn, task_id: str, kind: str) -> list:
    return [event for event in kb.list_events(conn, task_id) if event.kind == kind]


def _runs(conn, task_id: str) -> list[dict]:
    """The card's attempt history: each of its task_runs rows, oldest first."""
    return [
        dict(run)
        for run in conn.execute(
            "SELECT * FROM task_runs WHERE task_id = ? ORDER BY id", (task_id,),
        )
    ]


def _assert_held(conn, task_id: str) -> None:
    """The card, never claimed, waits blocked ``needs_input``; no run exists."""
    row = conn.execute(
        "SELECT status, block_kind, current_run_id, started_at, claim_lock "
        "FROM tasks WHERE id = ?",
        (task_id,),
    ).fetchone()
    assert tuple(row) == ("blocked", "needs_input", None, None, None)
    assert _events(conn, task_id, "claimed") == []
    assert _events(conn, task_id, "blocked")[-1].payload["reason"] == oni.HOLD_SENTENCE
    # Unlike block_task on a card never claimed, the hold writes no run row,
    # not even a closed one of no length, however often the card is held.
    assert _runs(conn, task_id) == []


# ---------------------------------------------------------------------------
# The hold at the claim
# ---------------------------------------------------------------------------


def test_a_ready_card_missing_an_input_is_held_and_no_run_starts(ctx, fired_hooks):
    setup, task_id = _late_card(ctx, _PRICE_LIST)
    with contextlib.closing(kb.connect(board=setup["board"])) as conn:
        fired_hooks.store = _store_file(conn)
        assert _status(conn, task_id) == "ready"
        assert _runs(conn, task_id) == []

        assert kb.claim_task(conn, task_id) is None

        _assert_held(conn, task_id)
        assert _runs(conn, task_id) == []
        (blocked,) = _events(conn, task_id, "blocked")
        assert blocked.payload == {
            "reason": oni.HOLD_SENTENCE,
            "kind": "needs_input",
            "recurrences": 1,
            "source_status": "ready",
        }
        # There is no run, so the event names none.
        assert blocked.run_id is None
        assignee = kb.get_task(conn, task_id).assignee
    # Fired once, as block_task fires it, and after the commit: another
    # connection already saw the card blocked. No run id is invented.
    assert fired_hooks.fired == [(
        "kanban_task_blocked",
        {
            "task_id": task_id,
            "profile_name": get_active_profile_name(),
            "board": kb.get_current_board(),
            "assignee": assignee,
            "run_id": None,
            "reason": oni.HOLD_SENTENCE,
        },
        "blocked",
    )]


def test_the_same_card_is_claimed_once_the_owner_gave_each_file(ctx, fired_hooks):
    setup, task_id = _late_card(ctx, _PRICE_LIST, _OPENING_HOURS)
    board = setup["board"]
    with contextlib.closing(kb.connect(board=board)) as conn:
        fired_hooks.store = _store_file(conn)
        assert kb.claim_task(conn, task_id) is None
        _assert_held(conn, task_id)

        # One file given: the card is held again for the other, and this hold
        # adds no run row either.
        _give(conn, board, task_id, "price-list.csv")
        assert kb.unblock_task(conn, task_id) is True
        assert kb.claim_task(conn, task_id) is None
        _assert_held(conn, task_id)

        # Each file given under its derived name: the card is claimed.
        _give(conn, board, task_id, "opening-hours.txt")
        assert kb.unblock_task(conn, task_id) is True
        claimed = kb.claim_task(conn, task_id)

        assert claimed is not None and claimed.status == "running"
        assert _status(conn, task_id) == "running"
        # Its one run is the claimed one; the two holds left none behind.
        (run,) = _runs(conn, task_id)
        assert (run["id"], run["status"], run["ended_at"]) == (
            claimed.current_run_id, "running", None,
        )
        assert [event.payload["recurrences"] for event in _events(conn, task_id, "blocked")] == [1, 2]
        assert len(_events(conn, task_id, "claimed")) == 1
    assert [(hook, status) for hook, _fields, status in fired_hooks.fired] == [
        ("kanban_task_blocked", "blocked"),
        ("kanban_task_blocked", "blocked"),
        ("kanban_task_claimed", "running"),
    ]


# "default" is also the card's own assignee.
@pytest.mark.parametrize("uploader", ["agent", "dashboard", "default", None])
def test_a_file_uploaded_by_anyone_else_does_not_count(ctx, uploader):
    setup, task_id = _card(ctx, _PRICE_LIST)
    with contextlib.closing(kb.connect(board=setup["board"])) as conn:
        _give(conn, setup["board"], task_id, "price-list.csv", uploaded_by=uploader)

        assert kb.claim_task(conn, task_id) is None

        _assert_held(conn, task_id)


def test_a_card_without_a_declaration_is_claimed_as_today(ctx, fired_hooks):
    setup, task_id = _card(ctx)
    with contextlib.closing(kb.connect(board=setup["board"])) as conn:
        fired_hooks.store = _store_file(conn)

        claimed = kb.claim_task(conn, task_id)

        assert claimed is not None and claimed.status == "running"
        assert _events(conn, task_id, "blocked") == []
    assert [(hook, status) for hook, _fields, status in fired_hooks.fired] == [
        ("kanban_task_claimed", "running"),
    ]


def test_a_card_still_waiting_on_a_parent_is_not_held(ctx, fired_hooks):
    setup, task_id = _late_card(ctx, _PRICE_LIST)
    with contextlib.closing(kb.connect(board=setup["board"])) as conn:
        fired_hooks.store = _store_file(conn)
        parent = kb.create_task(
            conn, title="Collect the prices", assignee="default",
            project_id=setup["project_id"],
        )
        kb.link_tasks(conn, parent, task_id)
        # A racy writer put the card back in ready before its parent is done.
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (task_id,))

        assert kb.claim_task(conn, task_id) is None

        # The parents check answers first; the hold does not touch the card.
        assert _status(conn, task_id) == "todo"
        assert [event.payload for event in _events(conn, task_id, "claim_rejected")] == [
            {"reason": "parents_not_done"},
        ]
        assert _events(conn, task_id, "blocked") == []
    assert fired_hooks.fired == []


def test_a_running_card_is_never_held(ctx, fired_hooks):
    setup, task_id = _late_card(ctx, _PRICE_LIST)
    board = setup["board"]
    with contextlib.closing(kb.connect(board=board)) as conn:
        fired_hooks.store = _store_file(conn)
        _give(conn, board, task_id, "price-list.csv")
        assert kb.claim_task(conn, task_id) is not None
        # The owner's file is gone while the card runs; another claim of the
        # running card changes nothing.
        (given,) = kb.list_attachments(conn, task_id)
        kb.delete_attachment(conn, given.id)
        assert oni.missing_inputs(conn, task_id) != []

        assert kb.claim_task(conn, task_id) is None

        assert _status(conn, task_id) == "running"
        assert _events(conn, task_id, "blocked") == []
    assert [hook for hook, _fields, _status in fired_hooks.fired] == ["kanban_task_claimed"]


def test_block_task_still_records_one_closed_attempt_for_a_card_never_claimed(ctx):
    # Only the hold leaves the attempt history empty: block_task keeps the
    # default and keeps its reason there as one closed record of no length.
    setup, task_id = _card(ctx)
    reason = "Which venue should we book?"
    with contextlib.closing(kb.connect(board=setup["board"])) as conn:
        assert _runs(conn, task_id) == []

        assert kb.block_task(conn, task_id, reason=reason, kind="needs_input") is True

        (run,) = _runs(conn, task_id)
        assert (run["status"], run["outcome"], run["summary"]) == ("blocked", "blocked", reason)
        assert run["started_at"] == run["ended_at"]
        assert _events(conn, task_id, "blocked")[-1].run_id == run["id"]


# ---------------------------------------------------------------------------
# The Decision under owner_input_upload_v1
# ---------------------------------------------------------------------------


def test_each_type_has_one_plain_label_and_the_size_limit_is_8_mib():
    assert set(oni.TYPE_LABELS) == set(oni.INPUT_TYPES)
    assert oni.TYPE_LABELS["txt"] == "text file"
    assert oni.TYPE_LABELS["csv"] == "spreadsheet (CSV)"
    labels = list(oni.TYPE_LABELS.values())
    assert len(set(labels)) == len(labels)
    for label in labels:
        assert "." not in label
        for kind in ("title", "stop_reason_decision"):
            assert check_owner_text(label, kind) == label
    assert oni.MAX_INPUT_BYTES == 8 * 1024 * 1024


def test_with_the_capability_the_decision_lists_only_the_missing_inputs(ctx):
    setup, task_id = _card(ctx, _PRICE_LIST, _OPENING_HOURS, _MENU)
    board = setup["board"]
    with contextlib.closing(kb.connect(board=board)) as conn:
        _give(conn, board, task_id, "opening-hours.txt")
        assert kb.claim_task(conn, task_id) is None

    (today,) = ow.list_owner_decisions(ctx)["data"]
    (granted,) = ow.list_owner_decisions(ctx, owner_input_upload=True)["data"]

    assert set(today) == _DECISION_KEYS
    assert today["kind"] == "owner_input"
    assert {key: value for key, value in granted.items() if key != "missing_inputs"} == today
    # In declaration order, each with its plain label and the size limit.
    assert granted["missing_inputs"] == [
        {"n": 1, "name": "price list", "type_label": "spreadsheet (CSV)",
         "max_bytes": 8 * 1024 * 1024},
        {"n": 3, "name": "menu", "type_label": "formatted text (Markdown)",
         "max_bytes": 8 * 1024 * 1024},
    ]
    # Never a file name: nothing in the list carries a dot.
    assert "." not in json.dumps(granted["missing_inputs"])

    # Every file given: the card still waits for its release, missing none.
    with contextlib.closing(kb.connect(board=board)) as conn:
        _give(conn, board, task_id, "price-list.csv")
        _give(conn, board, task_id, "menu.md")
    (granted,) = ow.list_owner_decisions(ctx, owner_input_upload=True)["data"]
    assert granted["missing_inputs"] == []


def test_a_needs_input_card_without_a_declaration_gets_no_missing_inputs(ctx):
    setup, held = _card(ctx, _PRICE_LIST)
    with contextlib.closing(kb.connect(board=setup["board"])) as conn:
        assert kb.claim_task(conn, held) is None
        question = kb.create_task(
            conn, title="Choose the venue", assignee="raphael-worker",
            project_id=setup["project_id"],
        )
        assert kb.claim_task(conn, question) is not None
        assert kb.block_task(
            conn, question, reason="Which venue should we book?", kind="needs_input",
        ) is True

    granted = {
        item["title"]: item
        for item in ow.list_owner_decisions(ctx, owner_input_upload=True)["data"]
    }

    assert set(granted) == {"Publish the price page", "Choose the venue"}
    assert set(granted["Choose the venue"]) == _DECISION_KEYS
    assert [
        entry["name"] for entry in granted["Publish the price page"]["missing_inputs"]
    ] == ["price list"]


@pytest.mark.asyncio
async def test_only_a_reader_naming_the_capability_gets_missing_inputs(ctx, owner):
    token = ow.OWNER_INPUT_UPLOAD_CAPABILITY
    assert token == "owner_input_upload_v1"
    assert (owner.actor, owner.profile) == (ctx.actor, ctx.profile)
    setup, task_id = _card(ctx, _PRICE_LIST)
    with contextlib.closing(kb.connect(board=setup["board"])) as conn:
        assert kb.claim_task(conn, task_id) is None

    async with TestClient(TestServer(_app())) as client:
        today = await _get(client, _DECISIONS)
        granted = await _get(client, _DECISIONS, token)
        combined = await _get(client, _DECISIONS, f"{ow.OWNER_WAITING_CAPABILITY},{token}")
        refused = [
            await _get(client, _DECISIONS, *values)
            for values in ((token, token), ("owner_input_upload_v2",), ("",))
        ]

    (item,) = today["data"]
    assert set(item) == _DECISION_KEYS
    for answer in (granted, combined):
        (listed,) = answer["data"]
        assert listed["missing_inputs"] == [{
            "n": 1, "name": "price list", "type_label": "spreadsheet (CSV)",
            "max_bytes": oni.MAX_INPUT_BYTES,
        }]
    (listed,) = granted["data"]
    assert {key: value for key, value in listed.items() if key != "missing_inputs"} == item
    # A repeated parameter or an unknown token is today's answer exactly.
    for answer in refused:
        assert answer == today


# ---------------------------------------------------------------------------
# The hold at the activation (card P1)
# ---------------------------------------------------------------------------

_TITLE = "Publish the price page"
# The parking of the one approval whose cards the tests below release.
_GENERATION = kb.park_generation(actor="owner", profile="default", idempotency_key="card-p1")


def _park(conn, task_id: str, *inputs: dict) -> str:
    """Park *task_id* under :data:`_GENERATION`, as an approval parks its work.

    The card declares *inputs* through the real declaration; without any it
    declares nothing.
    """
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET status = ?, park_generation = ? WHERE id = ?",
            (kb.PARKED_STATUS, _GENERATION, task_id),
        )
        if inputs:
            oni.declare_named_inputs(
                conn, [{"title": _TITLE, "inputs": list(inputs)}], [task_id],
            )
    return task_id


def _parked(conn, project_id: str, *inputs: dict, parents: tuple = ()) -> str:
    """One new card of *project_id*, parked by :func:`_park`."""
    task_id = kb.create_task(
        conn, title=_TITLE, assignee="default", project_id=project_id, parents=parents,
    )
    return _park(conn, task_id, *inputs)


def _parent(conn, project_id: str) -> str:
    """A parent card that is not done."""
    return kb.create_task(
        conn, title="Collect the prices", assignee="default", project_id=project_id,
    )


def _state(conn, task_id: str) -> tuple[dict, list[str], list[dict]]:
    """The card's row, the kinds of its events and its runs."""
    row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
    return dict(row), [event.kind for event in kb.list_events(conn, task_id)], _runs(conn, task_id)


def test_activation_holds_card_with_missing_input(ctx):
    setup, task_id = _card(ctx, _PRICE_LIST)
    with contextlib.closing(kb.connect(board=setup["board"])) as conn:
        # Right after the approval, never claimed: held, and no run exists.
        _assert_held(conn, task_id)
        (blocked,) = _events(conn, task_id, "blocked")
        # The claim's hold records exactly this.
        assert blocked.payload == {
            "reason": oni.HOLD_SENTENCE,
            "kind": "needs_input",
            "recurrences": 1,
            "source_status": "ready",
        }
        assert blocked.run_id is None
        kinds = [event.kind for event in kb.list_events(conn, task_id)]
        assert kinds.index("owner_work_activated") < kinds.index("blocked")
    # So the Decisions page offers the Upload at once.
    (item,) = ow.list_owner_decisions(ctx, owner_input_upload=True)["data"]
    assert item["title"] == _TITLE
    assert item["missing_inputs"] == [{
        "n": 1, "name": "price list", "type_label": "spreadsheet (CSV)",
        "max_bytes": oni.MAX_INPUT_BYTES,
    }]


def test_activation_leaves_card_without_declaration(ctx):
    setup, task_id = _card(ctx)
    with contextlib.closing(kb.connect(board=setup["board"])) as conn:
        assert _status(conn, task_id) == "ready"
        assert _events(conn, task_id, "blocked") == []
    assert ow.list_owner_decisions(ctx, owner_input_upload=True)["data"] == []


def test_activation_leaves_card_with_each_file_given(ctx, fired_hooks):
    setup = _bootstrap_board(ctx)
    board = setup["board"]
    with contextlib.closing(kb.connect(board=board)) as conn:
        fired_hooks.store = _store_file(conn)
        task_id = _parked(conn, setup["project_id"], _PRICE_LIST, _OPENING_HOURS)
        _give(conn, board, task_id, "price-list.csv")
        _give(conn, board, task_id, "opening-hours.txt")

        assert kb.activate_owner_work(conn, [task_id], generation=_GENERATION) == [task_id]

        assert _status(conn, task_id) == "ready"
        assert _events(conn, task_id, "blocked") == []
    assert fired_hooks.fired == []


def test_activation_leaves_restored_blocked_dependent(ctx, fired_hooks):
    setup = _bootstrap_board(ctx)
    with contextlib.closing(kb.connect(board=setup["board"])) as conn:
        # A dependent a person blocked, which the plan parked out of blocked;
        # it declares a file the owner has not given.
        dependent = kb.create_task(
            conn, title=_TITLE, assignee="default", project_id=setup["project_id"],
        )
        assert kb.block_task(
            conn, dependent, reason="Which venue should we book?", kind="needs_input",
        ) is True
        _park(conn, dependent, _PRICE_LIST)
        added = _parked(conn, setup["project_id"], _PRICE_LIST)
        fired_hooks.store = _store_file(conn)
        row, kinds, runs = _state(conn, dependent)

        released = kb.activate_owner_work(
            conn, [dependent, added], generation=_GENERATION,
            restore_statuses={dependent: "blocked"},
        )

        assert released == [dependent, added]
        # Back in blocked exactly as today: no hold, nothing else changed.
        assert _state(conn, dependent) == (
            {**row, "status": "blocked", "park_generation": None},
            [*kinds, "owner_work_activated"],
            runs,
        )
        # The card this same call moved to todo is held.
        _assert_held(conn, added)
    assert [fields["task_id"] for _hook, fields, _status in fired_hooks.fired] == [added]


def test_activation_replay_holds_once(ctx, fired_hooks):
    setup = _bootstrap_board(ctx)
    with contextlib.closing(kb.connect(board=setup["board"])) as conn:
        fired_hooks.store = _store_file(conn)
        held = _parked(conn, setup["project_id"], _PRICE_LIST)
        waiting = _parked(
            conn, setup["project_id"], _MENU, parents=(_parent(conn, setup["project_id"]),),
        )
        assert kb.activate_owner_work(
            conn, [held, waiting], generation=_GENERATION,
        ) == [held, waiting]
        # A person releases the second without its file: todo, behind its parent.
        assert kb.unblock_task(conn, waiting) is True
        assert _status(conn, waiting) == "todo"
        before = [_state(conn, task_id) for task_id in (held, waiting)]

        # The same approval's activation again, as a replay of its receipt runs it.
        assert kb.activate_owner_work(conn, [held, waiting], generation=_GENERATION) == []

        # It holds and releases nothing.
        assert [_state(conn, task_id) for task_id in (held, waiting)] == before
        _assert_held(conn, held)
        for task_id in (held, waiting):
            assert [event.payload for event in _events(conn, task_id, "owner_work_activated")] == [None]
    assert [(hook, fields["task_id"]) for hook, fields, _status in fired_hooks.fired] == [
        ("kanban_task_blocked", held),
        ("kanban_task_blocked", waiting),
    ]


def test_activation_holds_card_behind_a_parent(ctx):
    setup = _bootstrap_board(ctx)
    board = setup["board"]
    with contextlib.closing(kb.connect(board=board)) as conn:
        parent = _parent(conn, setup["project_id"])
        task_id = _parked(conn, setup["project_id"], _PRICE_LIST, parents=(parent,))

        assert kb.activate_owner_work(conn, [task_id], generation=_GENERATION) == [task_id]

        # Waiting for its file at once, although its parent is not done.
        _assert_held(conn, task_id)
        assert _status(conn, parent) != "done"
        # The Upload of its last missing file releases it behind its parent.
        assert oni.accept_upload(
            conn, task_id, 1, b"item,price\ntea,3\n", "text/csv", board=board,
        ) == "released"
        assert _status(conn, task_id) == "todo"


def test_upload_releases_card_held_at_activation(ctx):
    setup, task_id = _card(ctx, _PRICE_LIST, _OPENING_HOURS)
    board = setup["board"]
    with contextlib.closing(kb.connect(board=board)) as conn:
        _assert_held(conn, task_id)
        assert oni.accept_upload(
            conn, task_id, 1, b"item,price\ntea,3\n", "text/csv", board=board,
        ) == "stored"
        _assert_held(conn, task_id)

        assert oni.accept_upload(
            conn, task_id, 2, b"Monday to Friday\n", "text/plain", board=board,
        ) == "released"

        assert _status(conn, task_id) == "ready"
        claimed = kb.claim_task(conn, task_id)
        assert claimed is not None and claimed.status == "running"


def test_claim_still_holds_a_card_missing_an_input(ctx, fired_hooks):
    setup, task_id = _card(ctx, _PRICE_LIST)
    with contextlib.closing(kb.connect(board=setup["board"])) as conn:
        fired_hooks.store = _store_file(conn)
        _assert_held(conn, task_id)
        # A person releases the card without its file.
        assert kb.unblock_task(conn, task_id) is True
        assert _status(conn, task_id) == "ready"

        assert kb.claim_task(conn, task_id) is None

        _assert_held(conn, task_id)
        assert [event.payload["recurrences"] for event in _events(conn, task_id, "blocked")] == [1, 2]
        assignee = kb.get_task(conn, task_id).assignee
    assert fired_hooks.fired == [(
        "kanban_task_blocked",
        {
            "task_id": task_id,
            "profile_name": get_active_profile_name(),
            "board": kb.get_current_board(),
            "assignee": assignee,
            "run_id": None,
            "reason": oni.HOLD_SENTENCE,
        },
        "blocked",
    )]


def test_activation_hook_fires_once_per_held_card_after_commit(ctx, fired_hooks):
    setup = _bootstrap_board(ctx)
    # The approval opens the Project's board without making it the current one.
    assert setup["board"] != kb.get_current_board()
    with contextlib.closing(kb.connect(board=setup["board"])) as conn:
        fired_hooks.store = _store_file(conn)
        first = _parked(conn, setup["project_id"], _PRICE_LIST)
        plain = _parked(conn, setup["project_id"])
        second = _parked(conn, setup["project_id"], _OPENING_HOURS, _MENU)

        # Forced to fail after both holds: the last card names no status to
        # return to, so the transaction rolls back.
        with pytest.raises(ValueError, match="cannot restore"):
            kb.activate_owner_work(
                conn, [first, second, plain], generation=_GENERATION,
                restore_statuses={plain: "nowhere"},
            )
        assert fired_hooks.fired == []
        for task_id in (first, second, plain):
            assert _status(conn, task_id) == kb.PARKED_STATUS
            assert _events(conn, task_id, "blocked") == []

        assert kb.activate_owner_work(
            conn, [first, plain, second], generation=_GENERATION,
        ) == [first, plain, second]
        assignee = kb.get_task(conn, first).assignee
    # Once for each held card, after the commit: another connection already
    # saw it blocked. A reader of the board each call names finds that block, once.
    assert fired_hooks.on_named_board == [("blocked", 1), ("blocked", 1)]
    assert fired_hooks.fired == [
        (
            "kanban_task_blocked",
            {
                "task_id": task_id,
                "profile_name": get_active_profile_name(),
                "board": setup["board"],
                "assignee": assignee,
                "run_id": None,
                "reason": oni.HOLD_SENTENCE,
            },
            "blocked",
        )
        for task_id in (first, second)
    ]
