"""``kanban_model_update_check``: the worker of a check card runs the check itself.

The owner's facts sheet of the card, points 14 and 15. On production the
worker of the practice card blocked it: the planning role has no command line,
so it could not run ``hermes model update-check``. The tool runs the check in
the worker's own process. It has no parameter: the card comes from
``HERMES_KANBAN_TASK``, a check card is known by its ``created_by`` alone, and
a practice card's role comes from its idempotency key.

Every test runs under the exact environment the dispatcher injects into a
worker, captured from the real spawn (``kanban_db._default_spawn``) with only
``Popen`` faked: ``HERMES_KANBAN_DB`` and ``HERMES_KANBAN_BOARD`` pinned to the
card's own board, ``HERMES_KANBAN_TASK`` and ``HERMES_KANBAN_RUN_ID`` set, and
``HERMES_HOME`` a profile home under the root. Where a test is about the
tool's own rule, ``run_update_check`` is a recording stand-in. Elsewhere the
check is the real one, with fake routes, sign-ins, account lists, capability
rows and probes as in ``tests/hermes_cli/test_model_update_check.py``, and its
proof is read again from the board store files by a separate connection.
"""

from __future__ import annotations

import contextlib
import copy
import importlib
import json
import logging
import os
import sqlite3
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

TOOL = "kanban_model_update_check"
MODULE = "tools.model_update_check_tool"
PRACTICE, MONTHLY = "model-update-practice", "model-update-check"
PLANNER, BUSINESS, VERIFIER = "raphael-planner", "raphael-business", "raphael-verifier"
BOARD, SHOP, GARDEN = "planning", "shop", "garden"
MONTH = "2026-11"
RECEIPT = "model_update_checked"
OPUS, OPUS_NEW, SOL = "claude-opus-5-5", "claude-opus-5-6", "gpt-6.1-sol"
EFFORTS = frozenset({"low", "medium", "high", "xhigh", "max"})
# A sign-in value the fake seams hold; it must never come back out of the tool.
LEAK = "fake-sign-in-value-never-shown"
# The planning role's tools on production: no terminal, so no command line.
PLANNER_CONFIG = (
    "platform_toolsets:\n  cli:\n    - kanban\n    - raphael_sandbox\n    - skills\n    - web\n"
)
# Reason sentences of the check's receipt.
FILED = "One suggestion was filed."
NOTHING = "No newer version passed all three checks."
SPENT = "This planning card already filed its one suggestion."
# Paths a receipt may name.
PATHS = (
    "/srv/hermes/profiles/raphael-planner/config.yaml",
    "/srv/hermes/kanban/boards/shop/kanban.db",
)
# What no answer of the tool may hold: model ids, provider names and paths.
UNSAID = (OPUS, OPUS_NEW, SOL, "anthropic", "openai", "codex", "claude", "gpt", *PATHS, "/", ".hermes")


# ---------------------------------------------------------------------------
# Copied from tests/hermes_cli/test_model_update_check.py (copied, not
# imported, so this file stands on its own).
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _every_log_record():
    """Every record any logger creates, at DEBUG: raw message, args and exception, not a formatted view."""
    records = []
    factory = logging.getLogRecordFactory()

    def capture(*args, **kwargs):
        record = factory(*args, **kwargs)
        records.append(record)
        return record

    loggers = [logging.getLogger()] + [
        found for found in logging.root.manager.loggerDict.values() if isinstance(found, logging.Logger)
    ]
    prior = [(found, found.level, found.disabled) for found in loggers]
    logging.setLogRecordFactory(capture)
    for found in loggers:
        found.setLevel(logging.DEBUG)
        found.disabled = False
    try:
        yield records
    finally:
        logging.setLogRecordFactory(factory)
        for found, level, disabled in prior:
            found.setLevel(level)
            found.disabled = disabled


def _raw_text(records):
    parts = []
    for record in records:
        parts += [str(record.msg), repr(record.args), str(record.exc_text), str(record.stack_info)]
        if record.exc_info:
            parts.append(logging.Formatter().formatException(record.exc_info))
        try:
            parts.append(record.getMessage())
        except Exception:  # noqa: BLE001 - a bad record still has its raw parts above
            pass
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# The root, the card and its worker
# ---------------------------------------------------------------------------

@pytest.fixture
def root(tmp_path, monkeypatch):
    """A fresh shared root: the default and planning boards, and the role homes under it."""
    from agent.delegation_context import DELEGATED_CHILD_ENV_MARKER
    from hermes_cli import kanban_db as kb

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    for key in [k for k in os.environ if k.startswith("HERMES_KANBAN_")]:
        monkeypatch.delenv(key)
    for key in ("HERMES_PROFILE", "HERMES_SESSION_ID", DELEGATED_CHILD_ENV_MARKER):
        monkeypatch.delenv(key, raising=False)

    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    kb.create_board(BOARD)
    for role in (PLANNER, VERIFIER):
        profile = home / "profiles" / role
        profile.mkdir(parents=True)
        (profile / "config.yaml").write_text(PLANNER_CONFIG, encoding="utf-8")
    return home


@pytest.fixture
def world(root):
    """Two more boards and the root registry: the card's project is on ``shop``.

    Only the root registry links the planning card to the board its suggestion
    is filed on.
    """
    from hermes_cli import kanban_db as kb
    from hermes_cli import projects_db

    for slug in (SHOP, GARDEN):
        kb.create_board(slug)
    with projects_db.connect_closing(root / "projects.db") as conn:
        shop = projects_db.create_project(conn, name="Shop", board_slug=SHOP)
        projects_db.create_project(conn, name="Garden", board_slug=GARDEN)
    return SimpleNamespace(root=root, shop=shop)


def _practice_key(role, month=MONTH):
    return f"model-update-practice:{role}:{month}"


def _card(*, created_by, key, project_id=None):
    """A card for the planning role on the planning board, as its creator makes it."""
    from hermes_cli import kanban_db as kb
    from hermes_cli.owner_model_updates import PRACTICE_CARD_TITLE

    with kb.connect_closing(board=BOARD) as conn:
        return kb.create_task(
            conn, title=PRACTICE_CARD_TITLE,
            body="Run `hermes model update-check` and finish with its receipt.",
            assignee=PLANNER, created_by=created_by, idempotency_key=key, board=BOARD,
            project_id=project_id,
        )


def _claimed_card(*, created_by, key, project_id=None):
    """The card, claimed by the dispatcher: it has a running run."""
    from hermes_cli import kanban_db as kb

    task_id = _card(created_by=created_by, key=key, project_id=project_id)
    with kb.connect_closing(board=BOARD) as conn:
        task = kb.claim_task(conn, task_id)
    assert task is not None and task.status == "running" and task.current_run_id is not None
    assert (task.created_by, task.idempotency_key) == (created_by, key)
    return task


def _spawn(task, workspace, monkeypatch):
    """Run the real dispatcher spawn with only Popen captured; return (argv, env)."""
    from hermes_cli import kanban_db as kb

    captured = {}

    class _Proc:
        pid = 4242

    def _popen(cmd, *args, **kwargs):
        kwargs["stdout"].close()  # the worker log the dispatcher opened for the child
        captured.update(cmd=list(cmd), env=dict(kwargs["env"]))
        return _Proc()

    with monkeypatch.context() as patch:
        patch.setattr(kb, "_resolve_hermes_argv", lambda: ["hermes"])
        patch.setattr(subprocess, "Popen", _popen)
        workspace.mkdir(parents=True, exist_ok=True)
        assert kb._default_spawn(task, str(workspace), board=BOARD) == 4242
    return captured["cmd"], captured["env"]


def _worker(root, tmp_path, monkeypatch, task):
    """Become the card's worker: exactly the env the dispatcher's spawn gives it.

    Returns the toolsets the spawn pins for the worker.
    """
    from agent.delegation_context import DELEGATED_CHILD_ENV_MARKER
    from hermes_cli import kanban_db as kb

    cmd, env = _spawn(task, tmp_path / f"workspace-{task.id}", monkeypatch)
    assert env["HERMES_KANBAN_TASK"] == task.id
    assert env["HERMES_KANBAN_RUN_ID"] == str(task.current_run_id)
    assert env["HERMES_KANBAN_BOARD"] == BOARD
    assert Path(env["HERMES_KANBAN_DB"]).resolve() == (kb.board_dir(BOARD) / "kanban.db").resolve()
    assert Path(env["HERMES_HOME"]).resolve() == (root / "profiles" / task.assignee).resolve()
    assert DELEGATED_CHILD_ENV_MARKER not in env
    for key in set(os.environ) - set(env):
        monkeypatch.delenv(key)
    for key, value in env.items():
        if os.environ.get(key) != value:
            monkeypatch.setenv(key, value)
    assert cmd.count("--toolsets") == 1, cmd
    return [name.strip() for name in cmd[cmd.index("--toolsets") + 1].split(",")]


# ---------------------------------------------------------------------------
# The tool, as the worker's model reaches it
# ---------------------------------------------------------------------------

def _registered():
    """The built-in discovery imports the new module by name; its entry is a gated kanban tool."""
    from tools.registry import discover_builtin_tools, registry

    assert MODULE in discover_builtin_tools()
    entry = registry.get_entry(TOOL)
    assert entry is not None and entry.toolset == "kanban" and entry.check_fn is not None
    return entry


def _definitions(pin, *, as_sent=False):
    """Name -> schema of every tool the worker's model can reach from its pin, read fresh.

    Every schema, before Tool Search folds any into its bridge; ``as_sent``
    gives the list the worker's agent sends its model, after Tool Search.
    """
    import model_tools
    from tools.registry import discover_builtin_tools, invalidate_check_fn_cache

    discover_builtin_tools()
    invalidate_check_fn_cache()
    model_tools._clear_tool_defs_cache()
    try:
        defs = model_tools.get_tool_definitions(
            enabled_toolsets=pin, quiet_mode=True, skip_tool_search_assembly=not as_sent,
        )
    finally:
        invalidate_check_fn_cache()
        model_tools._clear_tool_defs_cache()
    return {d["function"]["name"]: d["function"] for d in defs if "function" in d}


def _kanban_names():
    """Every kanban tool whose own check_fn lists it for this session."""
    from tools.registry import invalidate_check_fn_cache, registry
    from toolsets import resolve_toolset

    invalidate_check_fn_cache()
    try:
        schema = registry.get_definitions(set(resolve_toolset("kanban")), quiet=True)
    finally:
        invalidate_check_fn_cache()
    return {s["function"].get("name") for s in schema if "function" in s}


def _dispatch(args=None):
    """The handler, through the registry, with the arguments a model sent."""
    from tools.registry import registry

    return registry.dispatch(TOOL, {} if args is None else args)


def _call(pin):
    """One tool call as the worker's agent makes it."""
    import model_tools

    return model_tools.handle_function_call(TOOL, {}, enabled_toolsets=pin)


def _receipt(result, reason):
    """A receipt as the check returns it: it names models, providers, sign-ins and paths."""
    filed = None
    if result == "filed":
        filed = {
            "board": SHOP, "task_id": "t_suggestion", "target_profile": PLANNER,
            "subject_id": f"model-update:anthropic:{OPUS}:{OPUS_NEW}:{MONTH}",
        }
    return {
        "schema_version": 1, "month": MONTH, "practice": False, "practice_role": None,
        "result": result, "reason": reason, "boards_read": ["default", BOARD, SHOP],
        "unread_roles": [{"role": BUSINESS, "error": "OSError", "path": PATHS[0]}],
        "models": [
            {"provider": "anthropic", "model": OPUS, "roles": [PLANNER], "result": "candidates",
             "sign_ins": [{"sign_in": "gateway", "roles": [PLANNER]}], "store": PATHS[1]},
            {"provider": "openai-codex", "model": SOL, "roles": ["default"], "result": "no_newer_version"},
        ],
        "candidates": [{"provider": "anthropic", "from_model": OPUS, "to_model": OPUS_NEW,
                        "result": "passed" if filed else "failed"}],
        "filed": filed,
        "budget": {"planning_card": "t_card", "filed_before": []},
        "error": None,
    }


def _stand_in(monkeypatch, *, receipt=None, raises=None):
    """``run_update_check`` replaced by a recorder of each call; it returns ``receipt`` or raises."""
    muc = importlib.import_module("hermes_cli.model_update_check")
    calls = []

    def run_update_check(*, practice_role=None, seams=None):
        calls.append({"practice_role": practice_role, "seams": seams})
        if raises is not None:
            raise raises
        return copy.deepcopy(receipt or _receipt("nothing_filed", NOTHING))

    monkeypatch.setattr(muc, "run_update_check", run_update_check)
    return calls


class _Fake:
    """Fake routes, sign-ins, account lists, capability rows and probes for the real check.

    The planning role runs ``OPUS``; its gateway sign-in lists ``OPUS`` and
    ``OPUS_NEW``. ``fail`` names the one seam that raises an error echoing the
    sign-in value, as a provider may echo a request header.
    """

    def __init__(self, fail=None):
        self.fail = fail
        self.calls = []

    def seams(self, muc):
        def seam(name, *args):
            self.calls.append((name, *args))
            if name == self.fail:
                raise RuntimeError(f"{name} echoed the header Authorization: Bearer {LEAK}")

        def routes(role):
            seam("routes", role)
            return (muc.Route(PLANNER, "anthropic", OPUS, "max", "routine"),) if role == PLANNER else ()

        def sign_ins(role, provider):
            seam("sign_ins", role, provider)
            return ("gateway",), ""

        def account_models(provider, sign_in):
            seam("account_models", provider, sign_in)
            return (OPUS, OPUS_NEW)

        def capabilities(provider, model):
            return muc.Capability(reasoning=True, context_window=200_000, efforts=EFFORTS)

        def probe(provider, model, sign_in, effort, question):
            seam("probe", provider, model, sign_in)
            answer = muc.PROBE_ANSWER if question == muc.PROBE_QUESTION else "?"
            return muc.ProbeReply(text=answer, input_tokens=40, output_tokens=2)

        return muc.Seams(
            routes=routes,
            sign_ins=sign_ins,
            account_models=account_models,
            capabilities=capabilities,
            probe=probe,
            route_revision=lambda role: f"rev-{role}",
            now=lambda: datetime(2026, 11, 2, 9, 0, tzinfo=timezone.utc),
        )


def _real_check(monkeypatch, fake):
    """The real check, whose default seams are the fakes: the tool passes it none."""
    muc = importlib.import_module("hermes_cli.model_update_check")
    monkeypatch.setattr(muc, "default_seams", lambda: fake.seams(muc))
    return muc


# ---------------------------------------------------------------------------
# The proof, read from the store files by a separate connection
# ---------------------------------------------------------------------------

def _board_files(root):
    """Every board store on disk, found by globbing, not through kanban_db."""
    files = {"default": root / "kanban.db"}
    for path in sorted((root / "kanban" / "boards").glob("*/kanban.db")):
        files[path.parent.name] = path
    return files


def _rows(path, sql, *params):
    conn = sqlite3.connect(f"{Path(path).resolve().as_uri()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(row) for row in conn.execute(sql, params)]
    finally:
        conn.close()


def _state(files):
    """Every row of every table of every board."""
    state = {}
    for board, path in files.items():
        tables = [row["name"] for row in _rows(path, "SELECT name FROM sqlite_master WHERE type = 'table'")]
        state[board] = {
            table: sorted(repr(sorted(row.items())) for row in _rows(path, f'SELECT * FROM "{table}"'))
            for table in tables
        }
    return state


def _count(files):
    """Independent count per board: model update suggestions and receipts."""
    return {
        board: (
            _rows(path, "SELECT COUNT(*) AS n FROM tasks WHERE task_kind = 'recommendation' "
                        "AND recommendation_subject_id LIKE 'model-update:%'")[0]["n"],
            _rows(path, "SELECT COUNT(*) AS n FROM task_events WHERE kind = ?", RECEIPT)[0]["n"],
        )
        for board, path in files.items()
    }


def _suggestions(files):
    """Every model update suggestion on every board."""
    found = []
    for board, path in files.items():
        rows = _rows(path, "SELECT * FROM tasks WHERE task_kind = 'recommendation' "
                           "AND recommendation_subject_id LIKE 'model-update:%' ORDER BY rowid")
        found.extend({**row, "board": board} for row in rows)
    return found


def _receipts(files, task_id):
    """The receipts on the card, read straight from its board."""
    rows = _rows(files[BOARD], "SELECT run_id, payload FROM task_events WHERE task_id = ? AND kind = ? "
                               "ORDER BY id", task_id, RECEIPT)
    return [(row["run_id"], json.loads(row["payload"])) for row in rows]


def _dump(files):
    """Every task row and event on every board, as text."""
    return "\n".join(
        json.dumps(row, default=str)
        for path in files.values()
        for table in ("tasks", "task_events")
        for row in _rows(path, f"SELECT * FROM {table}")
    )


def _assert_hidden_and_refused(pin, calls, files):
    """Not offered, and its handler refuses a direct call: nothing runs and no board changes."""
    before = _state(files)
    assert TOOL not in _definitions(pin)
    assert TOOL not in _kanban_names()
    out = json.loads(_dispatch())
    assert out.get("ok") is not True and str(out.get("error", "")).startswith(f"{TOOL} refused"), out
    assert calls == []
    assert _state(files) == before


# ---------------------------------------------------------------------------
# Who gets the tool
# ---------------------------------------------------------------------------

def test_a_planning_worker_of_a_practice_card_gets_the_tool_beside_the_kanban_tools(root, tmp_path, monkeypatch):
    from toolsets import resolve_toolset

    practice = _claimed_card(created_by=PRACTICE, key=_practice_key("default"))
    ordinary = _claimed_card(created_by=PLANNER, key=None)
    offered = {}
    for name, task in (("practice", practice), ("ordinary", ordinary)):
        with monkeypatch.context() as worker:
            pin = _worker(root, tmp_path, worker, task)
            assert "kanban" in pin and "terminal" not in pin
            offered[name] = _definitions(pin)

    _registered()
    assert TOOL in resolve_toolset("kanban")
    practice_tools, ordinary_tools = set(offered["practice"]), set(offered["ordinary"])
    assert {"kanban_show", "kanban_complete", "kanban_block", TOOL} <= practice_tools
    # The one tool is all the planning role gains, and it gets no command line.
    assert practice_tools - ordinary_tools == {TOOL}
    assert ordinary_tools <= practice_tools
    assert not {"terminal", "process", "execute_code"} & practice_tools
    schema = offered["practice"][TOOL]
    assert not schema["parameters"].get("properties") and not schema["parameters"].get("required")
    assert "hermes model update-check" in schema["description"]


def test_the_worker_is_sent_the_tool_by_name_beside_the_kanban_tools(root, tmp_path, monkeypatch):
    """What the worker's agent sends its model, with Tool Search as the planning
    role has it (the default): Tool Search folds every deferrable tool behind its
    bridge and never a kanban tool. The worker must see this tool by name, as it
    sees kanban_complete, not have to search for it."""
    pin = _worker(root, tmp_path, monkeypatch, _claimed_card(created_by=PRACTICE, key=_practice_key(PLANNER)))

    _registered()
    sent = set(_definitions(pin, as_sent=True))
    assert {"kanban_show", "kanban_complete", "kanban_block", TOOL} <= sent, sorted(sent)


@pytest.mark.parametrize(
    "created_by", [PLANNER, "default", "project-review", f"{PRACTICE}-v2", f"{MONTHLY}-v2"],
)
def test_a_card_another_creator_made_gets_no_tool_even_with_a_practice_key(root, tmp_path, monkeypatch, created_by):
    """A role makes a card with a board tool, which writes the role's name into
    ``created_by``; it can choose the idempotency key, and the key grants nothing."""
    calls = _stand_in(monkeypatch)
    pin = _worker(root, tmp_path, monkeypatch, _claimed_card(created_by=created_by, key=_practice_key(PLANNER)))

    _registered()
    _assert_hidden_and_refused(pin, calls, _board_files(root))


def test_a_review_run_of_a_check_card_gets_no_tool(root, tmp_path, monkeypatch):
    from hermes_cli import kanban_db as kb

    calls = _stand_in(monkeypatch)
    task = _claimed_card(created_by=PRACTICE, key=_practice_key(PLANNER))
    with kb.connect_closing(board=BOARD) as conn:
        assert kb.request_review(
            conn, task.id, summary="The check ran.", reviewer=VERIFIER, expected_run_id=task.current_run_id,
        )
        review = kb.claim_review_task(conn, task.id, claimer=f"{VERIFIER}:1")
        assert review is not None and review.current_run_id not in (None, task.current_run_id)
        assert kb.run_claimed_from_review(conn, task.id, review.current_run_id)
    pin = _worker(root, tmp_path, monkeypatch, review)

    _registered()
    _assert_hidden_and_refused(pin, calls, _board_files(root))


@pytest.mark.parametrize(
    "session",
    ["no_task", "no_run_id", "another_run", "delegated_child", "delegated_child_process", "cron_in_worker"],
)
def test_a_session_the_dispatcher_did_not_start_gets_no_tool(root, tmp_path, monkeypatch, session):
    from agent.delegation_context import (
        DELEGATED_CHILD_ENV_MARKER,
        delegated_child_context,
        non_dispatcher_owned_context,
    )

    calls = _stand_in(monkeypatch)
    task = _claimed_card(created_by=PRACTICE, key=_practice_key(PLANNER))
    pin = _worker(root, tmp_path, monkeypatch, task)

    _registered()
    # Positive control: the card's own worker is offered the tool.
    assert TOOL in _definitions(pin)
    scope = contextlib.nullcontext()
    if session == "no_task":
        monkeypatch.delenv("HERMES_KANBAN_TASK")
    elif session == "no_run_id":
        monkeypatch.delenv("HERMES_KANBAN_RUN_ID")
    elif session == "another_run":
        monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(task.current_run_id + 1))
    elif session == "delegated_child":
        scope = delegated_child_context()
    elif session == "delegated_child_process":
        monkeypatch.setenv(DELEGATED_CHILD_ENV_MARKER, "1")
    else:
        scope = non_dispatcher_owned_context()
    with scope:
        _assert_hidden_and_refused(pin, calls, _board_files(root))


# ---------------------------------------------------------------------------
# The one call and its role
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("role", ["default", PLANNER, BUSINESS])
def test_a_practice_card_runs_the_check_once_with_the_role_of_its_key(root, tmp_path, monkeypatch, role):
    calls = _stand_in(monkeypatch)
    _worker(root, tmp_path, monkeypatch, _claimed_card(created_by=PRACTICE, key=_practice_key(role)))

    out = json.loads(_dispatch())

    assert calls == [{"practice_role": role, "seams": None}]
    assert out == {"ok": True, "result": "nothing_filed", "reason": NOTHING, "filed": False}


@pytest.mark.parametrize(
    "key", [None, f"{MONTHLY}:{MONTH}", _practice_key(PLANNER)], ids=["no_key", "monthly_key", "practice_key"],
)
def test_a_monthly_card_runs_the_check_once_with_no_role(root, tmp_path, monkeypatch, key):
    calls = _stand_in(monkeypatch)
    _worker(root, tmp_path, monkeypatch, _claimed_card(created_by=MONTHLY, key=key))

    out = json.loads(_dispatch())

    assert calls == [{"practice_role": None, "seams": None}]
    assert out == {"ok": True, "result": "nothing_filed", "reason": NOTHING, "filed": False}


@pytest.mark.parametrize(
    "key",
    [
        None,
        f"{PRACTICE}:{PLANNER}",
        f"{PRACTICE}:{PLANNER}:2026-13",
        f"{PRACTICE}:{PLANNER}:{MONTH}:0",
        f"{PRACTICE}::{MONTH}",
        f"{PRACTICE}:{PLANNER}:{MONTH}\n",
        f"{MONTHLY}:{PLANNER}:{MONTH}",
        f"{PRACTICE}:raphael-owner:{MONTH}",
        f"{PRACTICE}:{PLANNER.upper()}:{MONTH}",
    ],
    ids=["no_key", "no_month", "bad_month", "extra_part", "no_role", "trailing_newline", "other_prefix",
         "unknown_role", "unknown_case"],
)
def test_a_practice_card_with_a_bad_key_or_an_unknown_role_is_refused_and_runs_nothing(
    root, tmp_path, monkeypatch, key,
):
    calls = _stand_in(monkeypatch)
    _worker(root, tmp_path, monkeypatch, _claimed_card(created_by=PRACTICE, key=key))
    files = _board_files(root)
    before = _state(files)

    out = json.loads(_dispatch())

    assert out.get("ok") is not True and str(out.get("error", "")).startswith(f"{TOOL} refused"), out
    assert calls == []
    assert _state(files) == before


def test_no_value_the_model_sends_reaches_the_check(root, tmp_path, monkeypatch):
    calls = _stand_in(monkeypatch)
    other = _card(created_by=PRACTICE, key=_practice_key(BUSINESS))
    _worker(root, tmp_path, monkeypatch, _claimed_card(created_by=PRACTICE, key=_practice_key(PLANNER)))

    out = json.loads(_dispatch({
        "practice_role": BUSINESS, "role": BUSINESS, "practice": BUSINESS, "task_id": other,
        "seams": "fake", "month": "2027-01",
    }))

    assert calls == [{"practice_role": PLANNER, "seams": None}]
    assert out == {"ok": True, "result": "nothing_filed", "reason": NOTHING, "filed": False}


# ---------------------------------------------------------------------------
# The answer
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "result, reason, filed", [("filed", FILED, True), ("nothing_filed", NOTHING, False)], ids=["filed", "nothing"],
)
def test_the_answer_holds_the_result_the_reason_and_whether_one_was_filed_only(
    root, tmp_path, monkeypatch, result, reason, filed,
):
    receipt = _receipt(result, reason)
    named = json.dumps(receipt)
    assert all(name in named for name in (OPUS, OPUS_NEW, SOL, "anthropic", "openai-codex", *PATHS))
    _stand_in(monkeypatch, receipt=receipt)
    _worker(root, tmp_path, monkeypatch, _claimed_card(created_by=PRACTICE, key=_practice_key(PLANNER)))

    raw = _dispatch()

    assert json.loads(raw) == {"ok": True, "result": result, "reason": reason, "filed": filed}
    assert [name for name in UNSAID if name in raw] == []


@pytest.mark.parametrize("kind", ["UpdateCheckRefused", "RuntimeError", "KeyError"])
def test_a_refused_or_failed_check_answers_with_its_kind_only(root, tmp_path, monkeypatch, kind):
    muc = importlib.import_module("hermes_cli.model_update_check")
    detail = f"{LEAK} at {PATHS[0]} for {OPUS} of anthropic"
    error = {
        "UpdateCheckRefused": muc.UpdateCheckRefused(f"the planning card belongs to another profile: {detail}"),
        "RuntimeError": RuntimeError(f"the store answered {detail}"),
        "KeyError": KeyError(detail),
    }[kind]
    calls = _stand_in(monkeypatch, raises=error)
    _worker(root, tmp_path, monkeypatch, _claimed_card(created_by=PRACTICE, key=_practice_key(PLANNER)))

    with _every_log_record() as records:
        raw = _dispatch()

    assert json.loads(raw) == {"error": f"{TOOL} failed: {kind}"}
    # One call, and no retry.
    assert calls == [{"practice_role": PLANNER, "seams": None}]
    assert [name for name in (LEAK, *UNSAID) if name in raw] == []
    assert LEAK not in _raw_text(records)


def test_the_answer_has_an_upper_length(root, tmp_path, monkeypatch):
    too_long = " ".join(
        f'Sentence {n} says "why" the check filed nothing, with a résumé of each board.' for n in range(40)
    )
    _stand_in(monkeypatch, receipt=_receipt("nothing_filed", too_long))
    _worker(root, tmp_path, monkeypatch, _claimed_card(created_by=PRACTICE, key=_practice_key(PLANNER)))
    limit = importlib.import_module(MODULE).ANSWER_MAX_CHARS
    assert isinstance(limit, int) and len(json.dumps(too_long)) > limit > len(NOTHING)

    raw = _dispatch()

    assert len(raw) <= limit
    out = json.loads(raw)
    assert (out["ok"], out["result"], out["filed"]) == (True, "nothing_filed", False)
    assert out["reason"][:40] == too_long[:40] and len(out["reason"]) < len(too_long)


# ---------------------------------------------------------------------------
# The real check, under the dispatcher's env
# ---------------------------------------------------------------------------

def test_one_call_on_a_practice_card_writes_its_receipt_and_files_one_practice_suggestion(
    world, tmp_path, monkeypatch,
):
    """Pinned to the planning board, the one call still reaches the root registry
    (the card's project files on ``shop``) and every other board. Its proof is
    read against an independent count of each store on disk."""
    fake = _Fake()
    muc = _real_check(monkeypatch, fake)
    task = _claimed_card(created_by=PRACTICE, key=_practice_key(PLANNER), project_id=world.shop)
    pin = _worker(world.root, tmp_path, monkeypatch, task)
    files = _board_files(world.root)
    assert set(files) == {"default", BOARD, SHOP, GARDEN}
    before = _count(files)

    out = json.loads(_call(pin))

    assert out == {"ok": True, "result": "filed", "reason": FILED, "filed": True}
    after = _count(files)
    assert {board: (after[board][0] - before[board][0], after[board][1] - before[board][1]) for board in files} == {
        "default": (0, 0), BOARD: (0, 1), SHOP: (1, 0), GARDEN: (0, 0),
    }
    ((run_id, receipt),) = _receipts(files, task.id)
    assert run_id == task.current_run_id
    assert (receipt["result"], receipt["practice"], receipt["practice_role"]) == ("filed", True, PLANNER)
    assert sorted(receipt["boards_read"]) == sorted(files)
    (row,) = _suggestions(files)
    assert (row["board"], row["project_id"], row["id"]) == (SHOP, world.shop, receipt["filed"]["task_id"])
    assert row["recommendation_subject_id"] == f"model-update:anthropic:{OPUS}:{OPUS}:{MONTH}"
    assert muc.PRACTICE_SENTENCE in row["recommendation_rationale"]
    assert row["provenance_ref"] == f"kanban-task:{task.id}"
    assert [call[1:] for call in fake.calls if call[0] == "probe"] == [("anthropic", OPUS, "gateway")]


def test_a_second_call_in_the_same_run_files_no_second_suggestion(world, tmp_path, monkeypatch):
    fake = _Fake()
    _real_check(monkeypatch, fake)
    task = _claimed_card(created_by=PRACTICE, key=_practice_key(PLANNER), project_id=world.shop)
    pin = _worker(world.root, tmp_path, monkeypatch, task)
    files = _board_files(world.root)
    before = _count(files)

    first = json.loads(_call(pin))
    second = json.loads(_call(pin))

    assert first == {"ok": True, "result": "filed", "reason": FILED, "filed": True}
    assert second == {"ok": True, "result": "nothing_filed", "reason": SPENT, "filed": False}
    after = _count(files)
    assert {board: (after[board][0] - before[board][0], after[board][1] - before[board][1]) for board in files} == {
        "default": (0, 0), BOARD: (0, 2), SHOP: (1, 0), GARDEN: (0, 0),
    }
    assert [receipt["result"] for _run_id, receipt in _receipts(files, task.id)] == ["filed", "nothing_filed"]
    assert len([call for call in fake.calls if call[0] == "probe"]) == 1


@pytest.mark.parametrize("fail", ["sign_ins", "account_models", "probe"])
def test_a_sign_in_value_in_the_seams_is_in_no_answer_no_log_line_and_no_error_text(
    world, tmp_path, monkeypatch, fail,
):
    fake = _Fake(fail=fail)
    _real_check(monkeypatch, fake)
    task = _claimed_card(created_by=PRACTICE, key=_practice_key(PLANNER), project_id=world.shop)
    pin = _worker(world.root, tmp_path, monkeypatch, task)
    files = _board_files(world.root)

    with _every_log_record() as records:
        raw = _call(pin)

    # The value was in play: the seam that echoes it ran inside the check.
    assert fail in [call[0] for call in fake.calls]
    expected = {
        "sign_ins": {"error": f"{TOOL} failed: RuntimeError"},
        "account_models": {"ok": True, "result": "nothing_filed", "reason": NOTHING, "filed": False},
        "probe": {"ok": True, "result": "nothing_filed", "reason": NOTHING, "filed": False},
    }
    assert json.loads(raw) == expected[fail]
    assert LEAK not in raw
    assert records and LEAK not in _raw_text(records)
    # The check's own record of the run is on the card, and holds no value either.
    ((_run_id, receipt),) = _receipts(files, task.id)
    assert receipt["error"] == ("RuntimeError" if fail == "sign_ins" else None)
    assert LEAK not in _dump(files)
