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


@pytest.fixture
def fired_hooks():
    """Each kanban_task_blocked and kanban_task_claimed call, in order.

    The registry ``invoke_hook`` reads is patched and restored, as
    tests/hermes_cli/test_kanban_lifecycle_hooks.py does. Each entry also holds
    the card's status as a second, read-only connection to ``store`` (set by
    the test) saw it while the hook ran: a hook fired inside the transaction
    would still see the card as it was before.
    """
    manager = get_plugin_manager()
    saved = {name: list(callbacks) for name, callbacks in manager._hooks.items()}
    seen = SimpleNamespace(store=None, fired=[])

    def capture(hook: str, fields: dict) -> None:
        status = _observe(
            seen.store, "SELECT status FROM tasks WHERE id = ?", fields["task_id"],
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
    setup, task_id = _card(ctx, _PRICE_LIST)
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
    setup, task_id = _card(ctx, _PRICE_LIST, _OPENING_HOURS)
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
    setup, task_id = _card(ctx, _PRICE_LIST)
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
    setup, task_id = _card(ctx, _PRICE_LIST)
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
