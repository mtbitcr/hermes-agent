"""A plan can name the text files a task needs from the owner.

Card C1 of the named inputs plan. Every task a plan creates may carry an
optional ``inputs`` list of one to five entries: a ``name`` (1 to 40 lowercase
letters, digits, single spaces and hyphens, starting with a letter, unique on
the task, unchanged by ``check_owner_text(name, "title")``) and a ``type``
(txt, md, csv or json). ``hermes_cli.owner_named_inputs`` is the one copy of
these rules: the approval authority and the apply path both call it, and the
apply path records each card's inputs as one ``named_inputs_declared`` task
event. ``missing_inputs`` answers which declared inputs the owner has not
attached yet.

Absent means no inputs: a proposal without the field authorizes and applies
exactly the bytes it did before. No card is parked for a missing input here.
"""

from __future__ import annotations

import copy
import hashlib
import json
import sqlite3
import types
from pathlib import Path
from unittest.mock import patch

import pytest

from gateway.platforms.api_server import APIServerAdapter, ResponseStore
from hermes_cli import kanban_db as kb
from hermes_cli import owner_named_inputs as oni
from hermes_cli import owner_workspace as ow
from hermes_cli.owner_text import check_owner_text

# The owner-workspace harness (temporary home, auto-approver, configured
# provider, canonical task graph) is reused verbatim, so every commit here
# goes through the REAL kernel entry point.
from tests.hermes_cli.test_owner_workspace import (  # noqa: F401
    _bootstrap_board,
    _configured_provider,
    _task_graph_args,
    _with_approver,
    ctx,
)
from tools import owner_workspace_tools as owt
from tools.registry import registry

_FIVE = [
    {"name": "price list", "type": "csv"},
    {"name": "opening hours", "type": "txt"},
    {"name": "menu", "type": "md"},
    {"name": "q3-report", "type": "json"},
    {"name": "sales 2026", "type": "csv"},
]
_LONGEST = "weekly opening hours for the main branch"  # 40 characters


def _named(*names: str, kind: str = "csv") -> list[dict]:
    return [{"name": name, "type": kind} for name in names]


# ---------------------------------------------------------------------------
# The rules
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "inputs",
    [
        pytest.param(_FIVE[:1], id="one"),
        pytest.param(_FIVE, id="five-of-every-type"),
        pytest.param(_named(_LONGEST, kind="txt"), id="forty-characters"),
        # Distinct names whose files differ by type: menu-list.txt and
        # menu-list.json.
        pytest.param(
            [{"name": "menu list", "type": "txt"}, {"name": "menu-list", "type": "json"}],
            id="same-stem-other-types",
        ),
    ],
)
def test_one_to_five_named_inputs_are_accepted(inputs):
    assert oni.parse_named_inputs(copy.deepcopy(inputs)) == inputs


@pytest.mark.parametrize(
    "inputs",
    [
        pytest.param([], id="no-entries"),
        pytest.param(_FIVE + _named("stock"), id="six-entries"),
        pytest.param(_named("price list", "price list"), id="duplicate-name"),
        pytest.param(
            [{"name": "price list", "type": "csv"}, {"name": "price list", "type": "json"}],
            id="duplicate-name-other-type",
        ),
        pytest.param(_named("price list", "price-list"), id="same-file-name"),
        pytest.param(_named("prices.csv"), id="extension"),
        pytest.param(_named("prices/2026"), id="slash"),
        pytest.param(_named("v1.2"), id="dot"),
        pytest.param(_named("claude opus"), id="model-name-rewritten"),
        pytest.param(_named("deadbeef"), id="hex-word-rewritten"),
        pytest.param(_named("price list", kind="pdf"), id="unknown-type"),
        pytest.param(_named("price list", kind="CSV"), id="type-not-lowercase"),
        pytest.param({"name": "price list", "type": "csv"}, id="object-not-list"),
        pytest.param("price list", id="string-not-list"),
        pytest.param(None, id="null-not-list"),
        pytest.param(["price list"], id="entry-not-object"),
        pytest.param([{"name": "price list"}], id="no-type"),
        pytest.param([{"type": "csv"}], id="no-name"),
        pytest.param(
            [{"name": "price list", "type": "csv", "file": "prices.csv"}],
            id="extra-key",
        ),
        pytest.param([{"name": 7, "type": "csv"}], id="name-not-string"),
        pytest.param(_named(""), id="empty-name"),
        pytest.param(_named(_LONGEST + "s"), id="forty-one-characters"),
        pytest.param(_named("Price list"), id="uppercase"),
        pytest.param(_named("2026 sales"), id="starts-with-digit"),
        pytest.param(_named("-price list"), id="starts-with-hyphen"),
        pytest.param(_named("price  list"), id="double-space"),
        pytest.param(_named(" price list"), id="leading-space"),
        pytest.param(_named("price list "), id="trailing-space"),
        pytest.param(_named("price_list"), id="underscore"),
        pytest.param(_named("prix détaillé"), id="not-ascii"),
    ],
)
def test_a_malformed_inputs_field_is_refused(inputs):
    with pytest.raises(ValueError):
        oni.parse_named_inputs(inputs)


def test_the_refused_names_are_the_ones_the_owner_text_policy_rewrites():
    # Guards the two policy cases above: they are refused because
    # check_owner_text rewrites them, not because of the character rule.
    for name in ("claude opus", "deadbeef"):
        assert check_owner_text(name, "title") != name
    for entry in _FIVE:
        assert check_owner_text(entry["name"], "title") == entry["name"]


def test_the_file_name_is_derived_stably_and_survives_the_attachment_store():
    assert oni.derived_file_name("price list", "csv") == "price-list.csv"
    assert oni.derived_file_name("q3-report", "json") == "q3-report.json"
    for entry in _FIVE + _named(_LONGEST, kind="json"):
        file_name = oni.derived_file_name(entry["name"], entry["type"])
        assert file_name == oni.derived_file_name(entry["name"], entry["type"])
        assert kb._safe_attachment_name(file_name) == file_name


def test_the_hold_sentence_is_fixed_and_passes_the_owner_text_policy():
    assert oni.HOLD_SENTENCE == (
        "Raphael needs a file from you before this work can start."
    )
    assert check_owner_text(
        oni.HOLD_SENTENCE, "stop_reason_decision",
    ) == oni.HOLD_SENTENCE


def test_the_tool_schemas_offer_inputs_as_an_optional_field():
    graph_task = registry.get_entry("owner_task_graph_commit").schema[
        "parameters"]["properties"]["tasks"]["items"]
    plan_change = registry.get_entry("owner_project_plan_commit").schema[
        "parameters"]["properties"]["changes"]["items"]["oneOf"][0]
    for spec in (
        graph_task,
        plan_change,
        owt._PROJECT_TASK_SPEC,
        owt._PROJECT_SPLIT_TASK_SPEC,
        owt._PROJECT_PRESERVED_TASK_SPEC,
        owt._PROJECT_REWRITTEN_TASK_SPEC,
    ):
        assert spec["properties"]["inputs"] is owt._NAMED_INPUTS
        assert "inputs" not in spec["required"]
    assert owt._NAMED_INPUTS["minItems"] == 1
    assert owt._NAMED_INPUTS["maxItems"] == 5
    entry = owt._NAMED_INPUTS["items"]
    assert entry["additionalProperties"] is False
    assert sorted(entry["required"]) == ["name", "type"]
    assert entry["properties"]["type"]["enum"] == ["txt", "md", "csv", "json"]
    assert entry["properties"]["name"]["maxLength"] == 40


# ---------------------------------------------------------------------------
# Through the real authority and apply path
# ---------------------------------------------------------------------------

_CONVERSATION = "raphael-owner-" + "5" * 32
_RESPONSE_ID = "resp_named_inputs_proposal"
_SECRET = "owner-executor-test-key"
_FIXED_PROJECT_ID = "project_raphael"
_PRICE_LIST = [{"name": "price list", "type": "csv"}]
_DECLARED_PRICE_LIST = {
    "inputs": [{"n": 1, "name": "price list", "type": "csv", "file": "price-list.csv"}],
}
# Measured on the code before this card: what a proposal without inputs
# authorized (the bound payload digest) and committed (the request digest)
# then. A proposal without inputs must keep exactly these bytes.
_TODAY_NEW_PROJECT_AUTHORITY_DIGEST = (
    "ff14e33a50f2fe4980a4bf22e7ac40e52be246fe06b5ad5fc15817abafc18060"
)
_TODAY_NEW_PROJECT_REQUEST_DIGEST = (
    "08aa4b1650682a8f361df32ad617d9baa01e8a61f9c2bba564081d2fef12d74b"
)
_TODAY_PLAN_CHANGE_AUTHORITY_DIGEST = (
    "df88ef27eed2e37ded89efa9c97b969bb1b659dd802207ef2a218ecf1bcccfb3"
)


def _idempotency_key() -> str:
    return "conversation-" + hashlib.sha256(_RESPONSE_ID.encode("utf-8")).hexdigest()


def _new_project(tasks: list[dict]) -> tuple[dict, dict, str]:
    """(stored proposal, run payload, operation) of a new Project."""
    payload = _task_graph_args(
        idempotency_key=_idempotency_key(), tasks=copy.deepcopy(tasks),
    )
    proposal = {
        "schema_version": 4,
        "kind": "proposal",
        "mode": "new",
        "project_name": payload["project_name"],
        "project_description": payload["project_description"],
        "request_title": payload["request_title"],
        "summary": "Prepare the first release.",
        "project_size": "small",
        "specification": payload["specification"],
        "current_milestone": payload["current_milestone"],
        "owner_visible_result": payload["owner_visible_result"],
        "impact": ["Adds one private Project."],
        "later_milestones": payload["later_milestones"],
        "tasks": copy.deepcopy(tasks),
    }
    return proposal, payload, "owner_task_graph_commit"


def _added_task(**extra) -> dict:
    return {
        "action": "add",
        "reason": "The release needs its price page.",
        "title": "Publish the price page",
        "body": "Publish the owner's current prices.",
        "assignee": "default",
        "responsibility": "B03",
        "execution_tier": "routine",
        "owned_paths": [],
        "risk_tier": 1,
        **extra,
    }


def _plan_change(project_id: str, **extra) -> tuple[dict, dict, str]:
    """(stored proposal, run payload, operation) of one ``add`` change."""
    proposal = {
        "schema_version": 6,
        "kind": "project_change_proposal",
        "mode": "existing",
        "request_title": "Add the price page",
        "summary": "Add one bounded task.",
        "project_size": "small",
        "specification": "Apply the approved change.",
        "current_milestone": "Publish the price page.",
        "owner_visible_result": "The owner sees the price page task.",
        "impact": ["Keeps completed work intact."],
        "later_milestones": [],
        "changes": [{
            **_added_task(**copy.deepcopy(extra)),
            "existing_parent_refs": [],
            "new_parents": [],
        }],
    }
    payload = {
        "idempotency_key": _idempotency_key(),
        "project_id": project_id,
        "trigger": "owner_request",
        "request_title": proposal["request_title"],
        "summary": proposal["summary"],
        "specification": proposal["specification"],
        "current_milestone": proposal["current_milestone"],
        "owner_visible_result": proposal["owner_visible_result"],
        "later_milestones": proposal["later_milestones"],
        "changes": [{
            **_added_task(**copy.deepcopy(extra)),
            "existing_parents": [],
            "new_parents": [],
        }],
    }
    return proposal, payload, "owner_project_plan_commit"


def _authorize(
    proposal: dict, payload: dict, operation: str,
    *, project_id: str = _FIXED_PROJECT_ID,
) -> dict:
    """Store *proposal* as the conversation's final reply; bind one run."""
    new = operation == "owner_task_graph_commit"
    authority = {
        "proposal_profile": "default",
        "conversation": _CONVERSATION,
        "response_id": _RESPONSE_ID,
        "claim_id": "claim_" + "7" * 32,
        "operation": operation,
        "idempotency_key": _idempotency_key(),
        "payload": payload,
    }
    context = (
        {
            "profile": "default", "mode": "new", "project_slug": None,
            "project_name": proposal.get("project_name"),
        }
        if new
        else {
            "profile": "default", "mode": "existing",
            "project_slug": "board-setup", "project_name": "Board Setup",
        }
    )
    snapshot = {
        "project": {"id": project_id},
        "planning_context": {
            "schema_version": 1,
            "actionable_count": 1,
            "omitted_terminal_count": 0,
            "actionable_truncated": False,
            "relations_truncated": False,
            "tasks": [{
                "id": "task_anchor", "title": "Board Setup", "status": "todo",
                "event_revision": 1, "parent_ids": [], "child_ids": [],
                "omitted_parent_count": 0, "omitted_child_count": 0,
            }],
        },
    }
    store = ResponseStore(max_size=10)
    adapter = APIServerAdapter.__new__(APIServerAdapter)
    adapter._response_store = store
    adapter._expected_api_key = lambda: _SECRET
    try:
        store.put(_RESPONSE_ID, {
            "response": {"id": _RESPONSE_ID, "created_at": 1},
            "conversation_history": [
                {"role": "user", "content": "Plan the work."},
                {"role": "assistant", "content": json.dumps(proposal)},
            ],
        })
        store.set_conversation(_CONVERSATION, _RESPONSE_ID, owner_proposal=True)
        with (
            patch(
                "hermes_cli.owner_workspace.resolve_owner_context",
                return_value=object(),
            ),
            patch(
                "hermes_cli.owner_workspace.read_project_snapshot",
                return_value=snapshot,
            ),
            patch(
                "hermes_cli.profiles.list_profiles",
                return_value=[types.SimpleNamespace(name="default")],
            ),
        ):
            return adapter._validated_owner_proposal_authority(
                authority, context, "default",
            )
    finally:
        store.close()


def _apply(ctx, binding: dict, *, approve: bool = True) -> dict:
    """Run the bound payload through the real apply path, as the gateway does."""
    function = {
        "owner_task_graph_commit": ow.commit_task_graph,
        "owner_project_plan_commit": ow.commit_project_plan,
    }[binding["operation"]]
    authorized = ow.OwnerContext(
        actor=ctx.actor,
        profile=ctx.profile,
        session=ctx.session,
        authority=ow.OwnerProposalAuthority(
            actor=ctx.actor,
            profile=ctx.profile,
            session=ctx.session,
            conversation=binding["conversation"],
            response_id=binding["response_id"],
            operation=binding["operation"],
            idempotency_key=binding["idempotency_key"],
            payload_digest=binding["payload_digest"],
        ),
    )
    approver = _with_approver(ctx.session) if approve else None
    try:
        return function(authorized, **copy.deepcopy(binding["payload"]))
    finally:
        if approver is not None:
            approver.join()


def _events(board: str, task_id: str, kind: str) -> list:
    conn = kb.connect(board=board)
    try:
        return [
            event.payload
            for event in kb.list_events(conn, task_id)
            if event.kind == kind
        ]
    finally:
        conn.close()


def _status(board: str, task_id: str) -> str:
    conn = kb.connect(board=board)
    try:
        return kb.get_task(conn, task_id).status
    finally:
        conn.close()


def test_a_new_project_with_inputs_declares_them_once_on_its_card(ctx):
    tasks = _task_graph_args()["tasks"]
    tasks[0]["inputs"] = copy.deepcopy(_PRICE_LIST)
    binding = _authorize(*_new_project(tasks))
    assert binding["payload"]["tasks"][0]["inputs"] == _PRICE_LIST

    result = _apply(ctx, binding)

    assert result["ok"] is True
    first, second = result["task_ids"]
    board = result["board"]
    assert _events(board, first, "named_inputs_declared") == [_DECLARED_PRICE_LIST]
    assert _events(board, second, "named_inputs_declared") == []
    # No card is parked for a missing input in this card: the declaring card
    # is activated exactly like its sibling.
    assert _status(board, first) not in {"blocked", kb.PARKED_STATUS}
    # A replay of the committed receipt declares nothing twice.
    assert _apply(ctx, binding, approve=False)["task_ids"] == [first, second]
    assert _events(board, first, "named_inputs_declared") == [_DECLARED_PRICE_LIST]


def test_a_plan_change_with_inputs_declares_them_once_on_its_card(ctx):
    setup = _bootstrap_board(ctx)
    binding = _authorize(
        *_plan_change(setup["project_id"], inputs=_PRICE_LIST),
        project_id=setup["project_id"],
    )
    assert binding["payload"]["changes"][0]["inputs"] == _PRICE_LIST

    result = _apply(ctx, binding)

    assert result["ok"] is True
    (created,) = result["created_task_ids"]
    assert _events(setup["board"], created, "named_inputs_declared") == [
        _DECLARED_PRICE_LIST,
    ]
    assert _status(setup["board"], created) not in {"blocked", kb.PARKED_STATUS}
    assert _apply(ctx, binding, approve=False)["created_task_ids"] == [created]
    assert _events(setup["board"], created, "named_inputs_declared") == [
        _DECLARED_PRICE_LIST,
    ]


def _declaring_binding(ctx, path: str) -> dict:
    """A run whose first created card declares the price list."""
    if path == "graph":
        tasks = _task_graph_args()["tasks"]
        tasks[0]["inputs"] = copy.deepcopy(_PRICE_LIST)
        return _authorize(*_new_project(tasks))
    setup = _bootstrap_board(ctx)
    return _authorize(
        *_plan_change(setup["project_id"], inputs=_PRICE_LIST),
        project_id=setup["project_id"],
    )


def _store_file(conn: sqlite3.Connection) -> str:
    return next(row[2] for row in conn.execute("PRAGMA database_list") if row[1] == "main")


def _observe(store_file: str, query: str, *params) -> int:
    """One count, as a second, read-only connection to the board sees it."""
    observer = sqlite3.connect(Path(store_file).as_uri() + "?mode=ro", uri=True)
    try:
        return observer.execute(query, params).fetchone()[0]
    finally:
        observer.close()


def _observed_card(store_file: str, task_id: str) -> tuple[int, int]:
    """(card, declaration) rows another connection sees for *task_id*."""
    return (
        _observe(store_file, "SELECT COUNT(*) FROM tasks WHERE id = ?", task_id),
        _observe(
            store_file,
            "SELECT COUNT(*) FROM task_events WHERE task_id = ? AND kind = ?",
            task_id, oni.DECLARED_EVENT,
        ),
    )


@pytest.mark.parametrize("path", ["graph", "plan"])
def test_no_observer_sees_a_card_without_its_declaration(ctx, path):
    binding = _declaring_binding(ctx, path)
    declare = ow.declare_named_inputs
    seen: dict = {}

    def observed_declare(conn, specs, task_ids):
        seen["store"], seen["card"] = _store_file(conn), task_ids[0]
        seen["in_transaction"] = conn.in_transaction
        seen["at_entry"] = _observed_card(seen["store"], seen["card"])
        declare(conn, specs, task_ids)
        seen["written"] = _observed_card(seen["store"], seen["card"])

    with patch("hermes_cli.owner_workspace.declare_named_inputs", observed_declare):
        assert _apply(ctx, binding)["ok"] is True

    # Declared inside the transaction that creates the card: until it
    # commits, another connection sees neither the card nor its declaration,
    # and afterwards both.
    assert seen["in_transaction"] is True
    assert seen["at_entry"] == seen["written"] == (0, 0)
    assert _observed_card(seen["store"], seen["card"]) == (1, 1)


@pytest.mark.parametrize("path", ["graph", "plan"])
def test_a_failed_declaration_write_leaves_neither_card_nor_declaration(ctx, path):
    binding = _declaring_binding(ctx, path)
    title = (binding["payload"].get("tasks") or binding["payload"]["changes"])[0]["title"]
    append_event = kb._append_event
    stores: list[str] = []

    def append_failing_declaration(conn, task_id, kind, payload=None, **kwargs):
        if kind == oni.DECLARED_EVENT:
            stores.append(_store_file(conn))
            raise sqlite3.OperationalError("disk I/O error")
        return append_event(conn, task_id, kind, payload, **kwargs)

    with patch("hermes_cli.kanban_db._append_event", append_failing_declaration):
        with pytest.raises(sqlite3.OperationalError, match="disk I/O error"):
            _apply(ctx, binding)

    (store,) = stores
    assert _observe(store, "SELECT COUNT(*) FROM tasks WHERE title = ?", title) == 0
    assert _observe(
        store, "SELECT COUNT(*) FROM task_events WHERE kind = ?", oni.DECLARED_EVENT,
    ) == 0


@pytest.mark.parametrize("inputs", [[], {"name": "price list", "type": "csv"}])
def test_a_proposal_with_malformed_inputs_authorizes_no_run(inputs):
    tasks = _task_graph_args()["tasks"]
    tasks[1]["inputs"] = inputs
    with pytest.raises(ValueError, match="stored proposal task is invalid"):
        _authorize(*_new_project(tasks))
    with pytest.raises(ValueError, match="stored proposal change is invalid"):
        _authorize(*_plan_change(_FIXED_PROJECT_ID, inputs=inputs))


@pytest.mark.parametrize("inputs", [_named("price list", "price-list"), "price list"])
def test_the_apply_path_refuses_malformed_inputs_before_anything_is_written(ctx, inputs):
    tasks = _task_graph_args()["tasks"]
    tasks[1]["inputs"] = inputs
    with pytest.raises(ow.OwnerWorkspaceError) as excinfo:
        ow._normalize_graph_tasks(tasks)
    assert excinfo.value.code == "invalid_argument"
    assert "tasks[1].inputs" in str(excinfo.value)
    with pytest.raises(ow.OwnerWorkspaceError) as excinfo:
        ow._normalize_project_changes([{
            **_added_task(inputs=inputs), "existing_parents": [], "new_parents": [],
        }])
    assert excinfo.value.code == "invalid_argument"
    assert "changes[0].inputs" in str(excinfo.value)


def test_a_new_project_without_inputs_is_applied_exactly_as_today(ctx):
    binding = _authorize(*_new_project(_task_graph_args()["tasks"]))
    assert "inputs" not in json.dumps(binding["payload"])
    assert binding["payload_digest"] == _TODAY_NEW_PROJECT_AUTHORITY_DIGEST

    result = _apply(ctx, binding)

    assert result["ok"] is True
    (decomposed,) = _events(result["board"], result["root_task_id"], "decomposed")
    assert decomposed["owner_task_graph_digest"] == _TODAY_NEW_PROJECT_REQUEST_DIGEST
    for task_id in result["task_ids"]:
        assert _events(result["board"], task_id, "named_inputs_declared") == []


def test_a_plan_change_without_inputs_is_applied_exactly_as_today(ctx):
    plan = _authorize(*_plan_change(_FIXED_PROJECT_ID))
    assert "inputs" not in json.dumps(plan["payload"])
    assert plan["payload_digest"] == _TODAY_PLAN_CHANGE_AUTHORITY_DIGEST
    setup = _bootstrap_board(ctx)
    applied = _apply(ctx, _authorize(
        *_plan_change(setup["project_id"]), project_id=setup["project_id"],
    ))
    assert applied["ok"] is True
    (created,) = applied["created_task_ids"]
    assert _events(setup["board"], created, "named_inputs_declared") == []
    conn = kb.connect(board=setup["board"])
    try:
        assert oni.missing_inputs(conn, created) == []
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# missing_inputs
# ---------------------------------------------------------------------------


def test_missing_inputs_counts_only_owner_attachments_with_the_derived_name(ctx):
    setup = _bootstrap_board(ctx)
    board = setup["board"]
    result = _apply(ctx, _authorize(
        *_plan_change(
            setup["project_id"],
            inputs=[
                {"name": "price list", "type": "csv"},
                {"name": "opening hours", "type": "txt"},
            ],
        ),
        project_id=setup["project_id"],
    ))
    (task_id,) = result["created_task_ids"]
    price = {"n": 1, "name": "price list", "type": "csv", "file": "price-list.csv"}
    hours = {"n": 2, "name": "opening hours", "type": "txt", "file": "opening-hours.txt"}

    conn = kb.connect(board=board)
    try:
        def attach(filename: str, uploaded_by: str) -> None:
            kb.store_attachment_bytes(
                conn, task_id, filename, b"data\n",
                uploaded_by=uploaded_by, board=board,
            )

        assert oni.missing_inputs(conn, task_id) == [price, hours]
        # An agent's file with the derived name does not count.
        attach("price-list.csv", "agent")
        assert oni.missing_inputs(conn, task_id) == [price, hours]
        # Nor does an owner file under any other name.
        attach("prices.csv", oni.OWNER_UPLOADER)
        attach("price list.csv", oni.OWNER_UPLOADER)
        assert oni.missing_inputs(conn, task_id) == [price, hours]
        # The owner's file with the derived name does, although the agent's
        # earlier file made the store keep it as "price-list (1).csv".
        attach("price-list.csv", oni.OWNER_UPLOADER)
        assert oni.missing_inputs(conn, task_id) == [hours]
        attach("opening-hours.txt", oni.OWNER_UPLOADER)
        assert oni.missing_inputs(conn, task_id) == []
    finally:
        conn.close()
