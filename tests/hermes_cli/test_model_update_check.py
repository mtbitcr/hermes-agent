"""Card 2 of the model update flow: ``hermes model update-check``.

For every model in use the check looks for a higher version of the same
family, runs the three checks (every sign-in lists it, the effort is
supported, the probe passes), files at most one suggestion through Decisions
and writes one receipt for each run.

Every run here happens under the exact environment the dispatcher injects into
a worker: it is captured from the real spawn with ``Popen`` faked. Routes,
sign-ins, account lists, capability rows and probes are fakes, and no sign-in
value exists anywhere in this file.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import importlib
import json
import logging
import os
import runpy
import shutil
import sqlite3
import subprocess
import sys
import threading
import tomllib
import warnings
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import fallback_config
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import projects_db
from hermes_cli.owner_text import check_owner_text
from plugins.dashboard_auth.raphael_workspace import model_policy

OPUS, OPUS_NEW, OPUS_NEWER = "claude-opus-5-5", "claude-opus-5-6", "claude-opus-5-7"
SONNET, SONNET_NEW = "claude-sonnet-5-5", "claude-sonnet-5-6"
SOL, SOL_NEW = "gpt-6.1-sol", "gpt-6.2-sol"
DEFAULT, PLANNER, BUSINESS = "default", "raphael-planner", "raphael-business"
DESIGNER, WORKER = "raphael-designer", "raphael-claude-worker"
MONTH = "2026-11"
RECEIPT = "model_update_checked"
# Text a fake sign-in reader or probe puts in its error; it must never be written.
LEAK = "fake-sign-in-value-never-written"
# Every effort a fake capability row accepts when it supports reasoning.
EFFORTS = frozenset({"low", "medium", "high", "xhigh", "max"})


def _muc():
    """The module under test, imported only after a test's setup has run."""
    return importlib.import_module("hermes_cli.model_update_check")


def _update(provider, old, new, month=MONTH):
    return f"model-update:{provider}:{old}:{new}:{month}"


def _at(year, month, day=2):
    return datetime(year, month, day, 9, 0, tzinfo=timezone.utc)


class Fake:
    """Fake routes, sign-ins, account lists, capability rows and probes."""

    def __init__(self, routes, lists, *, sign_ins=None, capabilities=None, answers=None):
        self.routes = [tuple(route) for route in routes]  # (role, provider, model, effort, tier)
        self.lists = dict(lists)  # (provider, sign-in name) -> ids, or the error the reader raises
        self.sign_ins = dict(sign_ins or {})  # role -> (sign-in names | None, reason)
        self.capabilities = dict(capabilities or {})  # model -> (reasoning, context window[, efforts])
        self.answers = dict(answers or {})  # model -> probe reply text, or the error the probe raises
        self.when = _at(2026, 11)
        self.list_calls = []
        self.probe_calls = []

    def seams(self, muc):
        def routes(role):
            return tuple(muc.Route(*route) for route in self.routes if route[0] == role)

        def sign_ins(role, provider):
            if role in self.sign_ins:
                return self.sign_ins[role]
            return ("gateway",) if provider == "anthropic" else ("codex",), ""

        def account_models(provider, sign_in):
            self.list_calls.append((provider, sign_in))
            listed = self.lists.get((provider, sign_in), ())
            if isinstance(listed, Exception):
                raise listed
            return tuple(listed)

        def capabilities(provider, model):
            # (reasoning, context window[, accepted efforts]); None is unknown.
            reasoning, context, *efforts = self.capabilities.get(model, (True, 200_000))
            if not efforts:
                efforts = [None if reasoning is None else EFFORTS if reasoning else frozenset()]
            return muc.Capability(reasoning=reasoning, context_window=context, efforts=efforts[0])

        def probe(provider, model, sign_in, effort, question):
            self.probe_calls.append(
                (provider, model, sign_in, effort, question, fallback_config.fallbacks_disabled())
            )
            right = muc.PROBE_ANSWER if question == muc.PROBE_QUESTION else "?"
            answer = self.answers.get(model, right)
            if isinstance(answer, Exception):
                raise answer
            return muc.ProbeReply(text=answer, input_tokens=40, output_tokens=2)

        return muc.Seams(
            routes=routes,
            sign_ins=sign_ins,
            account_models=account_models,
            capabilities=capabilities,
            probe=probe,
            route_revision=lambda role: f"rev-{role}",
            now=lambda: self.when,
        )


@dataclass
class World:
    root: Path
    card: str
    run_id: int
    shop: str
    garden: str
    env: dict
    after: dict = field(default_factory=dict)


def _dispatcher_env(task, workspace):
    """The env the dispatcher injects into this card's worker, from the real spawn."""
    captured = {}

    class _Proc:
        pid = 4242

    def _popen(cmd, *args, **kwargs):
        captured["env"] = dict(kwargs["env"])
        return _Proc()

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(subprocess, "Popen", _popen)
        kbd._default_spawn(task, str(workspace), board="planning")
    return captured["env"]


# The boards, the root registry and the unclaimed card, made once per module by
# the real APIs in the first test's home and copied into every later one. Making
# them costs seconds per test; the claim and the spawn below stay per test.
_TEMPLATE: dict = {}


@pytest.fixture
def world(fence_home, monkeypatch, tmp_path_factory):
    """Root home with four boards, a root registry and a claimed planning card.

    The card runs on the ``planning`` board; its project is registered to the
    ``shop`` board, so only the root registry links the two.
    """
    for var in ("HERMES_KANBAN_TASK", "HERMES_KANBAN_RUN_ID", "HERMES_PROFILE",
                fallback_config.FALLBACKS_DISABLED_ENV):
        monkeypatch.delenv(var, raising=False)
    root = fence_home
    if _TEMPLATE:
        shutil.copytree(_TEMPLATE["home"], root, dirs_exist_ok=True)
        shop, garden, card = _TEMPLATE["shop"], _TEMPLATE["garden"], _TEMPLATE["card"]
    else:
        kb.init_db()
        for slug in ("planning", "shop", "garden"):
            kb.create_board(slug)
        with projects_db.connect_closing(root / "projects.db") as conn:
            shop = projects_db.create_project(conn, name="Shop", board_slug="shop")
            garden = projects_db.create_project(conn, name="Garden", board_slug="garden")
        (root / "profiles" / PLANNER).mkdir(parents=True)
        with kb.connect_closing(board="planning") as conn:
            card = kb.create_task(conn, title="Monthly model check", assignee=PLANNER, project_id=shop)
        home = tmp_path_factory.mktemp("world") / "home"
        shutil.copytree(root, home, ignore=shutil.ignore_patterns("*.lock", "state.db*", "cache"))
        _TEMPLATE.update(home=home, shop=shop, garden=garden, card=card)
    with kb.connect_closing(board="planning") as conn:
        assert kb.get_task(conn, card).project_id == shop
        task = kb.claim_task(conn, card)
    assert task is not None and task.current_run_id is not None
    workspace = root / "kanban" / "boards" / "planning" / "workspaces" / card
    workspace.mkdir(parents=True)
    env = _dispatcher_env(task, workspace)
    assert env["HERMES_KANBAN_DB"] == str(root / "kanban" / "boards" / "planning" / "kanban.db")
    assert env["HERMES_KANBAN_BOARD"] == "planning"
    assert env["HERMES_KANBAN_TASK"] == card
    assert env["HERMES_KANBAN_RUN_ID"] == str(task.current_run_id)
    assert env["HERMES_HOME"] == str(root / "profiles" / PLANNER)
    return World(root=root, card=card, run_id=task.current_run_id, shop=shop, garden=garden, env=env)


def _run(world, fake, *, practice_role=None):
    """Run the check as the card's worker: the dispatcher's env and nothing else."""
    muc = _muc()
    seams = fake.seams(muc)
    with pytest.MonkeyPatch.context() as patch:
        for key in [key for key in os.environ if key not in world.env]:
            patch.delenv(key)
        for key, value in world.env.items():
            if os.environ.get(key) != value:
                patch.setenv(key, value)
        receipt = muc.run_update_check(practice_role=practice_role, seams=seams)
        world.after = {
            "env": dict(os.environ),
            "fallbacks_disabled": fallback_config.fallbacks_disabled(),
        }
    return receipt


def _board_files(root):
    """Every board database on disk, found by globbing, not through kanban_db."""
    files = {"default": root / "kanban.db"}
    for path in sorted((root / "kanban" / "boards").glob("*/kanban.db")):
        files[path.parent.name] = path
    return files


def _rows(path, sql, args=()):
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(row) for row in conn.execute(sql, args)]
    finally:
        conn.close()


def _suggestions(world):
    """Every model update suggestion on every board, in filing order per board."""
    found = []
    for board, path in _board_files(world.root).items():
        rows = _rows(
            path,
            "SELECT * FROM tasks WHERE task_kind = 'recommendation' "
            "AND recommendation_subject_id LIKE 'model-update:%' ORDER BY rowid",
        )
        found.extend({**row, "board": board} for row in rows)
    return found


def _receipts(world):
    """The receipts on the card, read straight from its board."""
    rows = _rows(
        _board_files(world.root)["planning"],
        "SELECT run_id, payload FROM task_events WHERE task_id = ? AND kind = ? ORDER BY id",
        (world.card, RECEIPT),
    )
    return [{"run_id": row["run_id"], "payload": json.loads(row["payload"])} for row in rows]


def _count(boards):
    """Independent count per board: model update suggestions and receipts."""
    return {
        board: (
            _rows(path, "SELECT COUNT(*) AS n FROM tasks WHERE task_kind = 'recommendation' "
                        "AND recommendation_subject_id LIKE 'model-update:%'")[0]["n"],
            _rows(path, "SELECT COUNT(*) AS n FROM task_events WHERE kind = ?", (RECEIPT,))[0]["n"],
        )
        for board, path in boards.items()
    }


def _dump(world):
    """Every task row and event on every board, as text."""
    parts = []
    for path in _board_files(world.root).values():
        for table in ("tasks", "task_events"):
            parts.extend(json.dumps(row, default=str) for row in _rows(path, f"SELECT * FROM {table}"))
    return "\n".join(parts)


def _files(root):
    """A hash of every file under the root home that is not a database."""
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file() and ".db" not in path.name
    }


_PAST_EVIDENCE = dict(
    schema_version=1,
    need="An earlier monthly check found a newer version for part of the team.",
    expected_benefit="New work could use the newer version.",
    requested_scope={flag: False for flag in kb.RECOMMENDATION_SCOPE_FLAGS},
    risks="The newer version may behave a little differently.",
    cost="Deciding costs nothing.",
    rollback="You can undo it in Settings.",
)


def _past_suggestion(world, board, subject, decision, *, project=None):
    """A suggestion an earlier run filed, decided by the owner on the Decisions screen."""
    with kb.connect_closing(board=board) as conn:
        task_id = kb.create_recommendation(
            conn,
            project_id=project or world.shop,
            target_profile=PLANNER,
            recommendation_kind="provider_model_policy",
            recommendation_subject_id=subject,
            recommendation_label="A newer version of an AI model is ready for part of your team.",
            recommendation_rationale="An earlier monthly check filed this.",
            recommendation_evidence=_PAST_EVIDENCE,
            provenance_authority=f"hermes-profile:{PLANNER}",
            provenance_ref="kanban-task:t_earlier",
        )
        kb.decide_recommendation_on_owner_screen(
            conn,
            task_id,
            decision=decision,
            reason="Owner choice on the Decisions screen.",
            actor="raphael-owner",
            owner_screen_ref="owner-workspace:decisions",
            expected_lifecycle_version=0,
        )
    return task_id


def test_higher_version_passing_all_three_checks_files_one_suggestion_and_one_receipt(world):
    fake = Fake(
        routes=[(PLANNER, "anthropic", OPUS, "max", "routine"), (DESIGNER, "anthropic", OPUS, "max", "routine")],
        lists={("anthropic", "gateway"): [SONNET, OPUS, OPUS_NEW]},
    )

    receipt = _run(world, fake)

    muc = _muc()
    (row,) = _suggestions(world)
    assert (row["board"], row["project_id"]) == ("shop", world.shop)
    assert row["recommendation_kind"] == "provider_model_policy"
    assert row["recommendation_subject_id"] == _update("anthropic", OPUS, OPUS_NEW)
    assert row["target_profile"] == PLANNER
    assert (row["status"], row["review_policy"]) == ("review", "owner")
    assert row["provenance_authority"] == f"hermes-profile:{PLANNER}"
    assert row["provenance_ref"] == f"kanban-task:{world.card}"

    assert _receipts(world) == [{"run_id": world.run_id, "payload": receipt}]
    assert (receipt["result"], receipt["practice"], receipt["month"]) == ("filed", False, MONTH)
    assert receipt["filed"]["task_id"] == row["id"]
    (checked,) = receipt["candidates"]
    assert (checked["provider"], checked["from_model"], checked["to_model"]) == ("anthropic", OPUS, OPUS_NEW)
    assert checked["result"] == "passed"
    assert checked["roles"] == [PLANNER, DESIGNER]
    assert checked["account"] == {"gateway": True}
    assert (checked["compatibility"]["effort"], checked["compatibility"]["context"]) == (True, True)
    assert [
        (probe["sign_in"], probe["passed"], probe["input_tokens"], probe["output_tokens"])
        for probe in checked["probe"]
    ] == [("gateway", True, 40, 2)]
    assert checked["route_revisions"] == {PLANNER: f"rev-{PLANNER}", DESIGNER: f"rev-{DESIGNER}"}
    assert [call[:5] for call in fake.probe_calls] == [
        ("anthropic", OPUS_NEW, "gateway", "max", muc.PROBE_QUESTION)
    ]
    assert str(world.root) not in json.dumps(row) + json.dumps(receipt)


@pytest.mark.parametrize(
    "build_list", [[SONNET], RuntimeError(f"401 for {LEAK}")], ids=["missing", "list-error"]
)
def test_candidate_one_sign_in_does_not_list_files_nothing(world, build_list):
    fake = Fake(
        routes=[(BUSINESS, "anthropic", SONNET, "high", "routine"), (WORKER, "anthropic", SONNET, "max", "routine")],
        lists={("anthropic", "gateway"): [SONNET, SONNET_NEW], ("anthropic", "build"): build_list},
        sign_ins={WORKER: (("build",), "")},
    )

    receipt = _run(world, fake)

    assert _suggestions(world) == []
    assert receipt["result"] == "nothing_filed"
    (checked,) = receipt["candidates"]
    assert (checked["to_model"], checked["result"]) == (SONNET_NEW, "not_listed")
    assert checked["account"] == {"gateway": True, "build": False}
    assert fake.probe_calls == []
    assert [event["payload"] for event in _receipts(world)] == [receipt]
    assert LEAK not in _dump(world)


def test_unsupported_effort_files_nothing(world):
    fake = Fake(
        routes=[(PLANNER, "anthropic", OPUS, "max", "routine")],
        lists={("anthropic", "gateway"): [OPUS, OPUS_NEW]},
        capabilities={OPUS_NEW: (False, 200_000)},
    )

    receipt = _run(world, fake)

    assert _suggestions(world) == []
    assert receipt["result"] == "nothing_filed"
    (checked,) = receipt["candidates"]
    assert checked["result"] == "effort_unsupported"
    assert checked["compatibility"]["effort"] is False
    assert fake.probe_calls == []
    assert len(_receipts(world)) == 1


@pytest.mark.parametrize(
    "answer", ["I would rather not say.", RuntimeError(f"401 for {LEAK}")], ids=["wrong-answer", "error"]
)
def test_failed_probe_files_nothing(world, caplog, answer):
    caplog.set_level(logging.DEBUG)
    fake = Fake(
        routes=[(PLANNER, "anthropic", OPUS, "max", "routine")],
        lists={("anthropic", "gateway"): [OPUS, OPUS_NEW]},
        answers={OPUS_NEW: answer},
    )

    receipt = _run(world, fake)

    assert _suggestions(world) == []
    assert receipt["result"] == "nothing_filed"
    (checked,) = receipt["candidates"]
    assert checked["result"] == "probe_failed"
    assert [probe["passed"] for probe in checked["probe"]] == [False]
    assert len(fake.probe_calls) == 1
    assert len(_receipts(world)) == 1
    assert LEAK not in _dump(world) + caplog.text


@pytest.mark.parametrize(
    ("routes", "listed", "winner"),
    [
        pytest.param(
            [(PLANNER, "anthropic", SONNET, "high", "routine"), (BUSINESS, "anthropic", OPUS, "max", "routine")],
            [OPUS_NEW, SONNET_NEW],
            (SONNET, SONNET_NEW),
            id="role-order",
        ),
        pytest.param(
            [(BUSINESS, "anthropic", OPUS, "high", "routine"), (BUSINESS, "anthropic", SONNET, "max", "deep")],
            [OPUS_NEW, SONNET_NEW],
            (SONNET, SONNET_NEW),
            id="deep-before-routine",
        ),
        pytest.param(
            [(PLANNER, "anthropic", OPUS, "max", "routine")],
            [OPUS_NEW, OPUS_NEWER],
            (OPUS, OPUS_NEWER),
            id="newest-first",
        ),
    ],
)
def test_two_candidates_file_one_in_rank_order(world, routes, listed, winner):
    fake = Fake(routes=routes, lists={("anthropic", "gateway"): [OPUS, SONNET, *listed]})

    receipt = _run(world, fake)

    (row,) = _suggestions(world)
    assert row["recommendation_subject_id"] == _update("anthropic", *winner)
    assert [checked["to_model"] for checked in receipt["candidates"]] == [winner[1]]
    assert [call[1] for call in fake.probe_calls] == [winner[1]]


def test_rejected_pair_is_never_filed_again(world):
    rejected = _update("anthropic", OPUS, OPUS_NEW, "2026-09")
    _past_suggestion(world, "shop", rejected, "rejected")
    fake = Fake(
        routes=[(PLANNER, "anthropic", OPUS, "max", "routine")],
        lists={("anthropic", "gateway"): [OPUS, OPUS_NEW]},
    )

    for when in (_at(2026, 11), _at(2027, 1)):
        fake.when = when
        receipt = _run(world, fake)
        assert receipt["result"] == "nothing_filed"
        assert [(c["to_model"], c["result"]) for c in receipt["candidates"]] == [(OPUS_NEW, "rejected_before")]

    assert [row["recommendation_subject_id"] for row in _suggestions(world)] == [rejected]
    assert fake.probe_calls == []
    assert len(_receipts(world)) == 2


def test_deferred_pair_is_filed_again_only_in_a_new_month(world):
    _past_suggestion(world, "shop", _update("anthropic", OPUS, OPUS_NEW), "deferred")
    fake = Fake(
        routes=[(PLANNER, "anthropic", OPUS, "max", "routine")],
        lists={("anthropic", "gateway"): [OPUS, OPUS_NEW]},
    )

    receipt = _run(world, fake)
    assert receipt["result"] == "nothing_filed"
    assert [(c["to_model"], c["result"]) for c in receipt["candidates"]] == [(OPUS_NEW, "deferred_this_month")]
    assert fake.probe_calls == []

    fake.when = _at(2026, 12, 1)
    receipt = _run(world, fake)
    assert receipt["result"] == "filed"
    assert [row["recommendation_subject_id"] for row in _suggestions(world)] == [
        _update("anthropic", OPUS, OPUS_NEW),
        _update("anthropic", OPUS, OPUS_NEW, "2026-12"),
    ]
    assert len(_receipts(world)) == 2


def test_owner_label_reason_and_evidence_pass_the_owner_text_check_and_name_no_model(world):
    fake = Fake(
        routes=[(PLANNER, "anthropic", OPUS, "max", "routine"), (BUSINESS, "anthropic", SONNET, "high", "routine")],
        lists={("anthropic", "gateway"): [OPUS, OPUS_NEW, SONNET]},
    )

    _run(world, fake)

    muc = _muc()
    (row,) = _suggestions(world)
    label, reason = row["recommendation_label"], row["recommendation_rationale"]
    evidence = json.loads(row["recommendation_evidence"])
    assert label == muc.OWNER_TITLE == "A newer version of an AI model is ready for part of your team."
    assert reason == muc.OWNER_REASON == (
        "Raphael checked that your account can use it, that it works with Raphael, and that it did one "
        "sample piece of work correctly. Update uses it for new work only. Work already started or "
        "waiting keeps its current setting. You can undo it in Settings."
    )
    assert check_owner_text(label, "title") == label
    assert check_owner_text(reason, "planner_prose") == reason
    texts = [evidence[key] for key in ("need", "expected_benefit", "risks", "cost", "rollback")]
    for text in texts:
        assert check_owner_text(text, "planner_prose") == text
    owner_text = " ".join([label, reason, *texts]).lower()
    for name in (OPUS, OPUS_NEW, SONNET, "anthropic", "openai", "codex", "claude", "opus", "sonnet", "gpt"):
        assert name not in owner_text


def test_probe_runs_with_fallbacks_disabled(world):
    assert fallback_config.FALLBACKS_DISABLED_ENV not in world.env
    fake = Fake(
        routes=[(BUSINESS, "anthropic", SONNET, "high", "routine"), (WORKER, "anthropic", SONNET, "max", "routine")],
        lists={("anthropic", "gateway"): [SONNET, SONNET_NEW], ("anthropic", "build"): [SONNET, SONNET_NEW]},
        sign_ins={WORKER: (("build",), "")},
    )

    receipt = _run(world, fake)

    assert receipt["result"] == "filed"
    # One short call for each sign-in, each one with fallbacks disabled.
    assert sorted(call[2] for call in fake.probe_calls) == ["build", "gateway"]
    assert [call[-1] for call in fake.probe_calls] == [True, True]
    # The run leaves the worker's process as it found it.
    assert fallback_config.FALLBACKS_DISABLED_ENV not in world.after["env"]
    assert world.after["fallbacks_disabled"] is False


def test_practice_files_a_suggestion_marked_as_practice_and_changes_no_route(world, monkeypatch):
    from hermes_cli import config as hermes_config

    def _refuse(*_args, **_kwargs):
        raise AssertionError("a practice run must not change a route, the configuration or the record")

    for owner, name in (
        (hermes_config, "apply_profile_route_batch"),
        (hermes_config, "save_config"),
        (model_policy, "apply_model_update_record"),
        (model_policy, "restore_model_update_record"),
    ):
        monkeypatch.setattr(owner, name, _refuse)
    fake = Fake(
        routes=[(PLANNER, "anthropic", OPUS, "max", "routine"), (BUSINESS, "anthropic", SONNET, "high", "routine")],
        lists={("anthropic", "gateway"): [OPUS, OPUS_NEW, SONNET]},
    )
    before = _files(world.root)

    receipt = _run(world, fake, practice_role=PLANNER)

    muc = _muc()
    assert _files(world.root) == before
    assert model_policy.read_model_updates() == ()
    (row,) = _suggestions(world)
    assert row["recommendation_subject_id"] == _update("anthropic", OPUS, OPUS)
    assert row["target_profile"] == PLANNER
    assert muc.PRACTICE_SENTENCE == "This is a practice run. Clicking Update changes nothing."
    assert muc.PRACTICE_SENTENCE in row["recommendation_rationale"]
    assert not any(json.loads(row["recommendation_evidence"])["requested_scope"].values())
    assert (receipt["result"], receipt["practice"]) == ("filed", True)
    assert receipt["filed"]["task_id"] == row["id"]
    # The probe still runs, on the role's current model.
    assert [call[:3] for call in fake.probe_calls] == [("anthropic", OPUS, "gateway")]


def test_unproven_sign_in_files_nothing_for_that_model_and_the_receipt_says_why(world):
    reason = "The worker uses the build sign-in or the gateway sign-in, and the check cannot tell which."
    fake = Fake(
        routes=[
            (PLANNER, "anthropic", OPUS, "max", "routine"),
            (BUSINESS, "anthropic", SONNET, "high", "routine"),
            (WORKER, "anthropic", OPUS, "max", "routine"),
        ],
        lists={
            ("anthropic", "gateway"): [OPUS, OPUS_NEW, SONNET, SONNET_NEW],
            ("anthropic", "build"): [OPUS, OPUS_NEW, SONNET, SONNET_NEW],
        },
        sign_ins={WORKER: (None, reason)},
    )

    receipt = _run(world, fake)

    assert [row["recommendation_subject_id"] for row in _suggestions(world)] == [
        _update("anthropic", SONNET, SONNET_NEW)
    ]
    (entry,) = [model for model in receipt["models"] if model["model"] == OPUS]
    assert entry["result"] == "sign_in_unproven"
    assert reason in entry["reason"]
    assert [call[1] for call in fake.probe_calls] == [SONNET_NEW]


def test_worker_env_reaches_every_board_and_the_root_registry(world):
    # Past decisions sit on boards the worker is not pinned to.
    _past_suggestion(world, "garden", _update("anthropic", OPUS, OPUS_NEW, "2026-08"), "rejected",
                     project=world.garden)
    _past_suggestion(world, "default", _update("openai-codex", SOL, SOL_NEW), "deferred")
    fake = Fake(
        routes=[
            (DEFAULT, "openai-codex", SOL, "max", "routine"),
            (PLANNER, "anthropic", OPUS, "max", "routine"),
            (BUSINESS, "anthropic", SONNET, "high", "routine"),
        ],
        lists={
            ("openai-codex", "codex"): [SOL, SOL_NEW],
            ("anthropic", "gateway"): [OPUS, OPUS_NEW, SONNET, SONNET_NEW],
        },
    )
    boards = _board_files(world.root)
    assert set(boards) == {"default", "planning", "shop", "garden"}
    before = _count(boards)

    receipt = _run(world, fake)

    after = _count(boards)
    # The receipt is the production proof; the counts above are read independently.
    assert sorted(receipt["boards_read"]) == sorted(boards)
    assert {board: after[board][0] - before[board][0] for board in boards} == {
        "default": 0, "planning": 0, "shop": 1, "garden": 0,
    }
    assert {board: after[board][1] - before[board][1] for board in boards} == {
        "default": 0, "planning": 1, "shop": 0, "garden": 0,
    }
    assert receipt["filed"]["board"] == "shop"
    assert receipt["filed"]["subject_id"] == _update("anthropic", SONNET, SONNET_NEW)
    assert [(c["to_model"], c["result"]) for c in receipt["candidates"]] == [
        (SOL_NEW, "deferred_this_month"),
        (OPUS_NEW, "rejected_before"),
        (SONNET_NEW, "passed"),
    ]
    assert [call[1] for call in fake.probe_calls] == [SONNET_NEW]


def test_update_check_command_runs_the_check_with_the_practice_role(monkeypatch):
    from hermes_cli.subcommands import model as model_subcommand

    muc = _muc()
    calls = []

    def run_update_check(*, practice_role=None, seams=None):
        calls.append(practice_role)
        return {"result": "nothing_filed", "reason": "No newer version was found.", "filed": None}

    monkeypatch.setattr(muc, "run_update_check", run_update_check)

    assert model_subcommand.cmd_model_update_check(argparse.Namespace(practice=None)) == 0
    assert model_subcommand.cmd_model_update_check(argparse.Namespace(practice=PLANNER)) == 0
    assert calls == [None, PLANNER]


# ---------------------------------------------------------------------------
# Review regressions: every run below goes through the command, in-process.
# ---------------------------------------------------------------------------


def _command(world, seams, *argv):
    """``hermes model update-check`` through the model parser and its handler, as the worker."""
    from hermes_cli.subcommands.model import build_model_parser

    muc = _muc()
    parser = argparse.ArgumentParser(prog="hermes")
    build_model_parser(parser.add_subparsers(dest="command"), cmd_model=lambda args: 0)
    with pytest.MonkeyPatch.context() as patch:
        for key in [key for key in os.environ if key not in world.env]:
            patch.delenv(key)
        for key, value in world.env.items():
            if os.environ.get(key) != value:
                patch.setenv(key, value)
        patch.setattr(muc, "default_seams", lambda: seams)
        args = parser.parse_args(["model", "update-check", *argv])
        assert args.func(args) == 0
    return _receipts(world)[-1]["payload"]


@pytest.mark.parametrize(
    "answer",
    [
        "The expression evaluates to 99, not 45.",
        "Not 45.",
        "It adds the whole numbers from 0 to 9.",
        "-45",
        "45.0",
    ],
    ids=["contradictory", "negated", "explanation-only", "near-negative", "near-decimal"],
)
def test_probe_answer_that_is_not_the_bare_number_files_nothing(world, answer):
    fake = Fake(
        routes=[(PLANNER, "anthropic", OPUS, "max", "routine")],
        lists={("anthropic", "gateway"): [OPUS, OPUS_NEW]},
        answers={OPUS_NEW: answer},
    )

    receipt = _command(world, fake.seams(_muc()))

    assert _suggestions(world) == []
    assert receipt["result"] == "nothing_filed"
    (checked,) = receipt["candidates"]
    assert checked["result"] == "probe_failed"
    assert [probe["passed"] for probe in checked["probe"]] == [False]


@pytest.mark.parametrize("answer", ["45", "45.", " 45 \n"], ids=["bare", "full-stop", "whitespace"])
def test_probe_answer_of_the_bare_number_passes(world, answer):
    fake = Fake(
        routes=[(PLANNER, "anthropic", OPUS, "max", "routine")],
        lists={("anthropic", "gateway"): [OPUS, OPUS_NEW]},
        answers={OPUS_NEW: answer},
    )

    receipt = _command(world, fake.seams(_muc()))

    assert receipt["result"] == "filed"
    assert [row["recommendation_subject_id"] for row in _suggestions(world)] == [
        _update("anthropic", OPUS, OPUS_NEW)
    ]


def test_probe_answer_format_is_the_whole_reply():
    muc = _muc()
    for text in ("45", "45.", "45!", "45\n", "  45  "):
        assert muc._answered(text), text
    for text in ("The expression evaluates to 99, not 45.", "Not 45", "45, not 99", "45 or 46",
                 "Answer: 45", "145", "450", "4.5", "45.5", "45.0", "-45", "forty-five", "", None):
        assert not muc._answered(text), text


def _catalog(monkeypatch, rows, contexts):
    """The catalog the default capability reader sees: ``rows`` models and ``contexts`` windows."""
    from agent import models_dev

    def capabilities(provider, model, **_kwargs):
        return models_dev.ModelCapabilities(supports_reasoning=True) if model in rows else None

    monkeypatch.setattr(models_dev, "get_model_capabilities", capabilities)
    monkeypatch.setattr(models_dev, "lookup_models_dev_context", lambda provider, model, **_: contexts.get(model))


@pytest.mark.parametrize(
    ("rows", "contexts", "result", "compatibility"),
    [
        pytest.param({OPUS}, {OPUS: 200_000}, "capability_unknown", (None, None), id="no-candidate-row"),
        pytest.param(set(), {}, "capability_unknown", (None, None), id="no-rows-at-all"),
        pytest.param({OPUS, OPUS_NEW}, {OPUS: 200_000}, "context_unknown", (True, None), id="no-candidate-context"),
        pytest.param({OPUS, OPUS_NEW}, {OPUS_NEW: 200_000}, "context_unknown", (True, None), id="no-current-context"),
    ],
)
def test_missing_compatibility_evidence_files_nothing_and_says_why(world, monkeypatch, rows, contexts,
                                                                   result, compatibility):
    muc = _muc()
    _catalog(monkeypatch, rows, contexts)
    fake = Fake(routes=[(PLANNER, "anthropic", OPUS, "max", "routine")],
                lists={("anthropic", "gateway"): [OPUS, OPUS_NEW]})

    receipt = _command(world, replace(fake.seams(muc), capabilities=muc._capability_row))

    assert _suggestions(world) == []
    assert receipt["result"] == "nothing_filed"
    (checked,) = receipt["candidates"]
    assert checked["result"] == result
    assert (checked["compatibility"]["effort"], checked["compatibility"]["context"]) == compatibility
    assert checked["compatibility"]["reason"]
    assert fake.probe_calls == []


def test_effort_the_role_uses_must_be_accepted_exactly(world):
    # The candidate reasons, but does not accept the exact effort the role runs at.
    fake = Fake(
        routes=[(PLANNER, "anthropic", OPUS, "max", "routine"), (DESIGNER, "anthropic", OPUS, "high", "deep")],
        lists={("anthropic", "gateway"): [OPUS, OPUS_NEW]},
        capabilities={OPUS_NEW: (True, 200_000, frozenset({"low", "medium", "high"}))},
    )

    receipt = _command(world, fake.seams(_muc()))

    assert _suggestions(world) == []
    (checked,) = receipt["candidates"]
    assert checked["result"] == "effort_unsupported"
    assert checked["compatibility"]["effort"] is False
    assert checked["compatibility"]["efforts"] == ["high", "max"]
    assert "max" in checked["compatibility"]["reason"]
    assert fake.probe_calls == []


def test_known_compatible_candidate_passes_through_the_default_reader(world, monkeypatch):
    muc = _muc()
    _catalog(monkeypatch, {OPUS, OPUS_NEW}, {OPUS: 200_000, OPUS_NEW: 1_000_000})
    fake = Fake(routes=[(PLANNER, "anthropic", OPUS, "max", "routine")],
                lists={("anthropic", "gateway"): [OPUS, OPUS_NEW]})

    receipt = _command(world, replace(fake.seams(muc), capabilities=muc._capability_row))

    assert receipt["result"] == "filed"
    (checked,) = receipt["candidates"]
    assert (checked["compatibility"]["effort"], checked["compatibility"]["context"]) == (True, True)
    assert [call[1:4] for call in fake.probe_calls] == [(OPUS_NEW, "gateway", "max")]


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


@pytest.mark.parametrize("provider", ["anthropic", "openai-codex"])
def test_default_account_list_error_never_logs_the_sign_in_value(world, monkeypatch, provider):
    import httpx

    from hermes_cli import models

    muc = _muc()
    monkeypatch.setattr(muc, "_sign_in_value", lambda *_: LEAK)

    def echoing(*_args, **kwargs):
        headers = kwargs.get("headers") or {}
        raise RuntimeError("Account response echoed " + " ".join(headers.values()))

    monkeypatch.setattr(models, "_get_json", echoing)
    monkeypatch.setattr(httpx, "get", echoing)
    role = (PLANNER, "anthropic", OPUS) if provider == "anthropic" else (DEFAULT, "openai-codex", SOL)
    fake = Fake(routes=[(*role, "max", "routine")], lists={})

    with _every_log_record() as records:
        receipt = _command(world, replace(fake.seams(muc), account_models=muc._account_model_ids))

    assert receipt["result"] == "nothing_filed"
    (entry,) = receipt["models"]
    assert entry["result"] == "list_unreadable"
    # The capture works: the check's own type-only line is there, the value is not.
    assert any("account list could not be read" in str(record.msg) for record in records)
    assert LEAK not in _raw_text(records)
    assert LEAK not in json.dumps(receipt) + _dump(world)


def test_repeated_run_under_one_planning_card_files_no_second_suggestion(world):
    fake = Fake(routes=[(PLANNER, "anthropic", OPUS, "max", "routine")],
                lists={("anthropic", "gateway"): [OPUS, OPUS_NEW, OPUS_NEWER]})
    boards = _board_files(world.root)
    before = _count(boards)

    first = _command(world, fake.seams(_muc()))
    second = _command(world, fake.seams(_muc()))

    after = _count(boards)
    assert [row["recommendation_subject_id"] for row in _suggestions(world)] == [
        _update("anthropic", OPUS, OPUS_NEWER)
    ]
    assert {board: after[board][0] - before[board][0] for board in boards} == {
        "default": 0, "planning": 0, "shop": 1, "garden": 0,
    }
    assert [event["payload"] for event in _receipts(world)] == [first, second]
    assert (first["result"], second["result"]) == ("filed", "nothing_filed")
    # The budget is checked before any further pair is chosen or probed.
    assert second["candidates"] == [] and second["models"] == []
    assert second["budget"]["filed_before"] == [
        {"board": "shop", "task_id": first["filed"]["task_id"], "subject_id": first["filed"]["subject_id"]}
    ]
    assert [call[1] for call in fake.probe_calls] == [OPUS_NEWER]


def test_concurrent_runs_under_one_planning_card_file_at_most_one(world, monkeypatch):
    from hermes_cli.subcommands.model import build_model_parser

    muc = _muc()
    barrier = threading.Barrier(2, timeout=60)
    seams = {}
    for name in ("monthly", "practice"):
        base = Fake(routes=[(PLANNER, "anthropic", OPUS, "max", "routine")],
                    lists={("anthropic", "gateway"): [OPUS, OPUS_NEWER]}).seams(muc)

        def probe(*args, _base=base):
            barrier.wait()  # both runs are past every check before either files
            return _base.probe(*args)

        seams[name] = replace(base, probe=probe)
    for key in [key for key in os.environ if key not in world.env]:
        monkeypatch.delenv(key)
    for key, value in world.env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(muc, "default_seams", lambda: seams[threading.current_thread().name])
    parser = argparse.ArgumentParser(prog="hermes")
    build_model_parser(parser.add_subparsers(dest="command"), cmd_model=lambda args: 0)
    codes = {}

    def command(name, *argv):
        args = parser.parse_args(["model", "update-check", *argv])
        codes[name] = args.func(args)

    threads = [threading.Thread(target=command, name="monthly", args=("monthly",)),
               threading.Thread(target=command, name="practice", args=("practice", "--practice", PLANNER))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(120)

    assert codes == {"monthly": 0, "practice": 0}
    assert len(_suggestions(world)) == 1
    receipts = [event["payload"] for event in _receipts(world)]
    assert sorted(receipt["result"] for receipt in receipts) == ["filed", "nothing_filed"]


# ---------------------------------------------------------------------------
# Second review: the real probe and each role's own sign-in, through the
# installed ``hermes`` entry point as the worker. Fakes sit only at the client
# and account boundaries; every value below is a placeholder, not a sign-in.
# ---------------------------------------------------------------------------


def _cli(world, seams, *argv, env=None):
    """``hermes model update-check`` through the installed entry point, as the worker.

    Where no ``hermes`` script is installed next to this Python, the target
    the script would call (``[project.scripts]`` in pyproject.toml) runs.
    """
    muc = _muc()
    env = dict(world.env if env is None else env)
    script = Path(sys.executable).parent / "hermes"
    with pytest.MonkeyPatch.context() as patch:
        for key in [key for key in os.environ if key not in env]:
            patch.delenv(key)
        for key, value in env.items():
            patch.setenv(key, value)
        patch.setattr(muc, "default_seams", lambda: seams)
        patch.setattr(sys, "argv", ["hermes", "model", "update-check", *argv])
        try:
            if script.exists():
                runpy.run_path(str(script), run_name="__main__")
            else:
                from hermes_cli.main import main

                sys.exit(main())
        except SystemExit as exc:
            assert exc.code in (None, 0), f"the command exited with {exc.code}"
    return _receipts(world)[-1]["payload"]


def _planner_pinned(world):
    """The planning profile runs on another model and the dispatcher's fallback switch is on."""
    (world.root / "profiles" / PLANNER / "config.yaml").write_text(json.dumps({
        "model": {"provider": "anthropic", "default": OPUS},
        "agent": {"reasoning_effort": "max"},
        "fallback_providers": [],
        "fallback_model": None,
    }))
    return {**world.env, fallback_config.FALLBACKS_DISABLED_ENV: "1"}


class _Http:
    """The fake HTTP boundary under the real provider SDKs: every request sent, and its answer.

    It answers each request in the format it asks for (a stream or one body),
    with 7 input and 1 output tokens. A refusal (``status``) echoes the sign-in
    the request carried, as a careless provider would. While a request is in
    flight the check's own clock moves ``after`` seconds, so no test waits.
    """

    def __init__(self, monkeypatch, *, status=200, after=1.0):
        import httpx

        self.requests = []
        self.status, self.after, self.now = status, after, 5000.0
        monkeypatch.setattr(httpx.Client, "send", lambda _client, request, **_kwargs: self.answer(request))
        monkeypatch.setattr(_muc(), "_clock", lambda: self.now, raising=False)

    def sent(self):
        return [json.loads(request.content) for request in self.requests]

    @staticmethod
    def credential(request):
        return request.headers.get("x-api-key") or request.headers.get("authorization", "").removeprefix("Bearer ")

    def answer(self, request):
        import httpx

        self.requests.append(request)
        self.now += self.after
        body = json.loads(request.content)
        if self.status != 200:
            error = {"type": "invalid_request_error",
                     "message": f"stream not supported for {self.credential(request)}"}
            return httpx.Response(self.status, json={"type": "error", "error": error}, request=request)
        if request.url.path.endswith("/responses"):
            item = {"type": "message", "id": "msg_fake", "role": "assistant", "status": "completed",
                    "content": [{"type": "output_text", "text": "45", "annotations": []}]}
            reply = {"id": "resp_fake", "object": "response", "created_at": 0, "status": "completed",
                     "model": body["model"], "output": [item],
                     "usage": {"input_tokens": 7, "output_tokens": 1, "total_tokens": 8}}
            events = [{"type": "response.output_item.done", "output_index": 0, "item": item},
                      {"type": "response.completed", "response": reply}]
            stream = "".join(f"data: {json.dumps(event)}\n\n" for event in events) + "data: [DONE]\n\n"
        else:
            reply = {"id": "msg_fake", "type": "message", "role": "assistant", "model": body["model"],
                     "content": [{"type": "text", "text": "45"}], "stop_reason": "end_turn",
                     "stop_sequence": None, "usage": {"input_tokens": 7, "output_tokens": 1}}
            events = [
                {"type": "message_start", "message": {**reply, "content": [], "stop_reason": None,
                                                      "usage": {"input_tokens": 7, "output_tokens": 0}}},
                {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
                {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "45"}},
                {"type": "content_block_stop", "index": 0},
                {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                 "usage": {"output_tokens": 1}},
                {"type": "message_stop"},
            ]
            stream = "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events)
        if body.get("stream"):
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=stream,
                                  request=request)
        return httpx.Response(200, json=reply, request=request)


def _probes(checked):
    return [(probe["sign_in"], probe["passed"], probe["input_tokens"], probe["output_tokens"])
            for probe in checked["probe"]]


def test_newer_candidate_probe_reaches_the_client_with_its_route_effort_and_sign_in(world, monkeypatch):
    muc = _muc()
    env = _planner_pinned(world)
    http = _Http(monkeypatch)
    monkeypatch.setattr(muc, "_sign_in_value", lambda provider, name: f"placeholder-{name}")
    fake = Fake(routes=[(PLANNER, "anthropic", OPUS, "max", "routine")],
                lists={("anthropic", "gateway"): [OPUS, OPUS_NEW]})

    receipt = _cli(world, replace(fake.seams(muc), probe=muc._probe_call), env=env)

    assert receipt["result"] == "filed"
    # One request, to the candidate, with that sign-in's exact value and the route's effort.
    assert [http.credential(request) for request in http.requests] == ["placeholder-gateway"]
    (sent,) = http.sent()
    assert sent["model"] == OPUS_NEW
    assert (sent["thinking"]["type"], sent["output_config"]) == ("adaptive", {"effort": "max"})
    (checked,) = receipt["candidates"]
    assert _probes(checked) == [("gateway", True, 7, 1)]
    assert "placeholder-gateway" not in json.dumps(receipt) + _dump(world)


def test_practice_probe_sends_each_sign_ins_exact_value_once_in_order(world, monkeypatch):
    muc = _muc()
    env = _planner_pinned(world)
    http = _Http(monkeypatch)
    monkeypatch.setattr(muc, "_sign_in_value", lambda provider, name: f"placeholder-{name}")
    fake = Fake(routes=[(WORKER, "anthropic", OPUS, "max", "deep")],
                lists={("anthropic", "gateway"): [OPUS], ("anthropic", "build"): [OPUS]},
                sign_ins={WORKER: (("gateway", "build"), "")})

    receipt = _cli(world, replace(fake.seams(muc), probe=muc._probe_call), "--practice", WORKER, env=env)

    assert receipt["result"] == "filed"
    assert [http.credential(request) for request in http.requests] == ["placeholder-gateway", "placeholder-build"]
    assert [sent["model"] for sent in http.sent()] == [OPUS, OPUS]
    assert _probes(receipt["candidates"][0]) == [("gateway", True, 7, 1), ("build", True, 7, 1)]


def test_codex_probe_builds_its_client_from_that_sign_ins_exact_value(world, monkeypatch):
    from agent import auxiliary_client as aux

    muc = _muc()
    env = _planner_pinned(world)
    http = _Http(monkeypatch)

    def refuse(*_args, **_kwargs):
        raise AssertionError("the probe never reads the pool, the stored login or the shared client cache")

    for name in ("_read_codex_access_token", "_select_pool_entry", "_peek_pool_entry", "_get_cached_client"):
        monkeypatch.setattr(aux, name, refuse)
    monkeypatch.setattr(muc, "_sign_in_value", lambda provider, name: f"placeholder-{name}")
    fake = Fake(routes=[(DEFAULT, "openai-codex", SOL, "max", "routine")],
                lists={("openai-codex", "codex"): [SOL, SOL_NEW]})

    receipt = _cli(world, replace(fake.seams(muc), probe=muc._probe_call), env=env)

    assert receipt["result"] == "filed"
    assert [http.credential(request) for request in http.requests] == ["placeholder-codex"]
    assert [(sent["model"], sent["stream"]) for sent in http.sent()] == [(SOL_NEW, True)]
    assert _probes(receipt["candidates"][0]) == [("codex", True, 7, 1)]


def _scoped_values(monkeypatch, module, name, answer):
    """Replace one sign-in resolver; it answers from the profile scope it runs in."""
    from hermes_constants import get_hermes_home

    scopes = []

    def resolver(*_args):
        scopes.append(get_hermes_home().name)
        return answer(get_hermes_home().name)

    monkeypatch.setattr(module, name, resolver)
    return scopes


def test_each_roles_own_gateway_account_must_list_the_candidate(world, monkeypatch):
    from agent import credential_pool

    from hermes_cli import models

    muc = _muc()
    values = {PLANNER: "placeholder-planner-account", DESIGNER: "placeholder-designer-account"}
    scopes = _scoped_values(
        monkeypatch, credential_pool, "preview_runtime_selection",
        lambda scope: (SimpleNamespace(runtime_api_key=values.get(scope), access_token=""), False),
    )
    listed = {values[PLANNER]: [OPUS, OPUS_NEW], values[DESIGNER]: [OPUS]}
    asked = []
    monkeypatch.setattr(models, "_fetch_anthropic_models",
                        lambda *, api_key, **_kwargs: asked.append(api_key) or listed[api_key])
    fake = Fake(routes=[(PLANNER, "anthropic", OPUS, "max", "routine"), (DESIGNER, "anthropic", OPUS, "max", "routine")],
                lists={})

    receipt = _cli(world, replace(fake.seams(muc), sign_ins=muc._role_sign_ins, account_models=muc._account_model_ids))

    assert receipt["result"] == "nothing_filed" and _suggestions(world) == []
    assert set(scopes) == {PLANNER, DESIGNER}
    (checked,) = receipt["candidates"]
    assert (checked["to_model"], checked["result"]) == (OPUS_NEW, "not_listed")
    assert sorted(checked["account"].values()) == [False, True]
    assert sorted(asked) == sorted(values.values())
    assert fake.probe_calls == []
    written = json.dumps(receipt) + _dump(world)
    assert not any(value in written for value in values.values())


def test_each_roles_own_codex_pool_account_must_list_the_candidate(world, monkeypatch):
    from agent import credential_pool

    from hermes_cli import codex_models

    muc = _muc()
    root = world.root.name
    values = {root: "placeholder-default-pool", BUSINESS: "placeholder-business-pool"}
    scopes = _scoped_values(
        monkeypatch, credential_pool, "preview_runtime_selection",
        lambda scope: (SimpleNamespace(runtime_api_key=values[scope], access_token=""), False),
    )
    listed = {values[root]: [SOL, SOL_NEW], values[BUSINESS]: [SOL]}
    monkeypatch.setattr(codex_models, "_fetch_models_from_api", lambda value: listed[value])
    fake = Fake(routes=[(DEFAULT, "openai-codex", SOL, "max", "routine"),
                        (BUSINESS, "openai-codex", SOL, "max", "routine")], lists={})

    receipt = _cli(world, replace(fake.seams(muc), sign_ins=muc._role_sign_ins, account_models=muc._account_model_ids))

    assert receipt["result"] == "nothing_filed" and _suggestions(world) == []
    assert set(scopes) == {root, BUSINESS}
    (checked,) = receipt["candidates"]
    assert (checked["to_model"], checked["result"]) == (SOL_NEW, "not_listed")
    assert sorted(checked["account"].values()) == [False, True]
    assert fake.probe_calls == []
    written = json.dumps(receipt) + _dump(world)
    assert not any(value in written for value in values.values())


@pytest.mark.parametrize("borrowed", [False, True], ids=["own-scope-empty", "only-planner-env"])
def test_role_whose_own_sign_in_cannot_be_proven_files_nothing_for_that_model(world, monkeypatch, borrowed):
    from agent import credential_pool
    from agent.secret_scope import get_secret

    from hermes_cli import models

    muc = _muc()
    # The planner and the business role have their own value; the designer has none.
    for role in (PLANNER, BUSINESS):
        home = world.root / "profiles" / role
        home.mkdir(parents=True, exist_ok=True)
        (home / ".env").write_text(f"ANTHROPIC_TOKEN=placeholder-{role}\n")
    env = dict(world.env)
    if borrowed:
        # The worker's own process environment holds the planner's value.
        env["ANTHROPIC_TOKEN"] = f"placeholder-{PLANNER}"
    else:
        env.pop("ANTHROPIC_TOKEN", None)
    scopes = _scoped_values(
        monkeypatch, credential_pool, "preview_runtime_selection",
        lambda _scope: (SimpleNamespace(runtime_api_key=get_secret("ANTHROPIC_TOKEN"), access_token=""), False),
    )
    asked = []
    monkeypatch.setattr(models, "_fetch_anthropic_models",
                        lambda *, api_key, **_kwargs: asked.append(api_key) or [OPUS, OPUS_NEW, SONNET, SONNET_NEW])
    fake = Fake(routes=[(PLANNER, "anthropic", OPUS, "max", "routine"),
                        (DESIGNER, "anthropic", OPUS, "max", "routine"),
                        (BUSINESS, "anthropic", SONNET, "high", "routine")], lists={})

    receipt = _cli(world, replace(fake.seams(muc), sign_ins=muc._role_sign_ins, account_models=muc._account_model_ids),
                   env=env)

    assert DESIGNER in scopes
    (opus,) = [entry for entry in receipt["models"] if entry["model"] == OPUS]
    assert (opus["result"], opus["role"]) == ("sign_in_unproven", DESIGNER)
    assert opus["reason"]
    # The other model still runs its checks, on its own role's account only.
    assert [row["recommendation_subject_id"] for row in _suggestions(world)] == [
        _update("anthropic", SONNET, SONNET_NEW)
    ]
    assert asked == [f"placeholder-{BUSINESS}"]
    assert [call[1] for call in fake.probe_calls] == [SONNET_NEW]
    written = json.dumps(receipt) + _dump(world)
    assert "placeholder-raphael" not in written and str(world.root) not in written


def test_every_proven_account_lists_the_candidate_files_with_one_list_and_probe_per_account(world, monkeypatch):
    from agent import credential_pool

    from hermes_cli import models

    muc = _muc()
    # The planner and the designer share one account; the business role has its own.
    values = {PLANNER: "placeholder-shared-account", DESIGNER: "placeholder-shared-account",
              BUSINESS: "placeholder-business-account"}
    scopes = _scoped_values(
        monkeypatch, credential_pool, "preview_runtime_selection",
        lambda scope: (SimpleNamespace(runtime_api_key=values.get(scope), access_token=""), False),
    )
    asked = []
    monkeypatch.setattr(models, "_fetch_anthropic_models",
                        lambda *, api_key, **_kwargs: asked.append(api_key) or [OPUS, OPUS_NEW])
    http = _Http(monkeypatch)
    fake = Fake(routes=[(role, "anthropic", OPUS, "max", "routine") for role in (PLANNER, DESIGNER, BUSINESS)],
                lists={})
    seams = replace(fake.seams(muc), sign_ins=muc._role_sign_ins, account_models=muc._account_model_ids,
                    probe=muc._probe_call)

    receipt = _cli(world, seams, env=_planner_pinned(world))

    assert receipt["result"] == "filed"
    assert set(scopes) == {PLANNER, DESIGNER, BUSINESS}
    distinct = sorted(set(values.values()))
    assert sorted(asked) == distinct
    assert sorted(http.credential(request) for request in http.requests) == distinct
    assert len(http.requests) == 2
    (entry,) = receipt["models"]
    assert sorted(sorted(group["roles"]) for group in entry["sign_ins"]) == [
        [BUSINESS], sorted([PLANNER, DESIGNER])
    ]
    (checked,) = receipt["candidates"]
    assert list(checked["account"].values()) == [True, True]
    assert [probe["passed"] for probe in checked["probe"]] == [True, True]
    written = json.dumps(receipt) + _dump(world)
    assert not any(value in written for value in values.values())


# ---------------------------------------------------------------------------
# The owner rule of 2026-11-17, contracts A to D, each as a whole class. The
# probe runs on the real provider SDKs; only the HTTP send is fake.
# ---------------------------------------------------------------------------


def _probed(provider):
    """One route whose newer candidate every sign-in lists, so only the probe decides."""
    if provider == "anthropic":
        return Fake(routes=[(PLANNER, "anthropic", OPUS, "max", "routine")],
                    lists={("anthropic", "gateway"): [OPUS, OPUS_NEW]})
    return Fake(routes=[(DEFAULT, "openai-codex", SOL, "max", "routine")],
                lists={("openai-codex", "codex"): [SOL, SOL_NEW]})


def _probe_run(world, monkeypatch, provider, value=None, **http):
    muc = _muc()
    env = _planner_pinned(world)
    wire = _Http(monkeypatch, **http)
    monkeypatch.setattr(muc, "_sign_in_value", lambda _provider, name: value or f"placeholder-{name}")
    with _every_log_record() as records:
        receipt = _cli(world, replace(_probed(provider).seams(muc), probe=muc._probe_call), env=env)
    return receipt, wire, records


@pytest.mark.parametrize("active_lists_it", [False, True], ids=["active-unlisted", "active-listed"])
def test_account_check_lists_the_account_the_roles_runtime_selects(world, monkeypatch, active_lists_it):
    """Contract A: the runtime takes the pool's choice before the legacy env resolver; so does the check."""
    from agent import credential_pool

    from hermes_cli import auth, models, runtime_provider

    muc = _muc()
    home = world.root / "profiles" / PLANNER
    active, other = "placeholder-active-pool-account", "placeholder-other-env-account"
    (home / ".env").write_text(f"ANTHROPIC_API_KEY={other}\n")
    row = {"id": "placeholder-active-row", "label": "placeholder active account", "auth_type": "api_key",
           "priority": 0, "source": "manual", "access_token": active, "base_url": "https://api.anthropic.com"}

    def in_planner_scope(reader):
        from agent.secret_scope import build_profile_secret_scope

        return muc._in_scope(home, build_profile_secret_scope(home), reader)

    store = []
    in_planner_scope(lambda: store.append(auth.write_credential_pool("anthropic", [row])) or "")
    # The real runtime and the read-only preview both run on the active pool account.
    assert in_planner_scope(lambda: runtime_provider.resolve_runtime_provider(
        requested="anthropic", target_model=OPUS)["api_key"]) == active
    assert in_planner_scope(lambda: credential_pool.preview_runtime_selection("anthropic")[0].runtime_api_key) == active
    before = store[0].read_bytes()
    asked = []

    def listed(*, api_key, **_kwargs):
        asked.append({active: "active", other: "other"}.get(api_key, "unexpected"))
        return [OPUS, OPUS_NEW] if (api_key == active) == active_lists_it else [OPUS]

    monkeypatch.setattr(models, "_fetch_anthropic_models", listed)
    fake = Fake(routes=[(PLANNER, "anthropic", OPUS, "max", "routine")], lists={})

    receipt = _cli(world, replace(fake.seams(muc), sign_ins=muc._role_sign_ins, account_models=muc._account_model_ids))

    assert asked == ["active"]
    assert receipt["result"] == ("filed" if active_lists_it else "nothing_filed")
    assert len(_suggestions(world)) == int(active_lists_it)
    assert [call[1] for call in fake.probe_calls] == ([OPUS_NEW] if active_lists_it else [])
    # Reading the choice wrote nothing to the store.
    assert store[0].read_bytes() == before
    written = json.dumps(receipt) + _dump(world)
    assert active not in written and other not in written


@pytest.mark.parametrize("provider", ["anthropic", "openai-codex"])
def test_role_whose_runtime_choice_cannot_be_read_files_nothing_for_that_model(world, monkeypatch, provider):
    """Contract A: no read of the choice, no proof; no other source stands in for it."""
    from agent import anthropic_credentials, credential_pool

    from hermes_cli import codex_models, models

    muc = _muc()

    def unreadable(_provider):
        raise PermissionError("the fake store cannot be read")

    monkeypatch.setattr(credential_pool, "preview_runtime_selection", unreadable)
    monkeypatch.setattr(anthropic_credentials, "resolve_anthropic_token", lambda *_: "placeholder-env-account")
    asked = []
    monkeypatch.setattr(models, "_fetch_anthropic_models",
                        lambda *, api_key, **_kwargs: asked.append(api_key) or [OPUS, OPUS_NEW])
    monkeypatch.setattr(codex_models, "_fetch_models_from_api", lambda value: asked.append(value) or [SOL, SOL_NEW])
    fake = Fake(routes=[(PLANNER, provider, OPUS if provider == "anthropic" else SOL, "max", "routine")], lists={})

    receipt = _cli(world, replace(fake.seams(muc), sign_ins=muc._role_sign_ins, account_models=muc._account_model_ids))

    (entry,) = receipt["models"]
    assert (entry["result"], entry["role"]) == ("sign_in_unproven", PLANNER)
    assert "PermissionError" in entry["reason"] and "fake store" not in json.dumps(receipt)
    assert receipt["result"] == "nothing_filed" and _suggestions(world) == []
    assert asked == [] and fake.probe_calls == []


@pytest.mark.parametrize("provider,status", [("anthropic", 400), ("anthropic", 529),
                                             ("openai-codex", 400), ("openai-codex", 500)])
def test_probe_refusal_is_one_request_and_a_failed_probe(world, monkeypatch, provider, status):
    """Contract B: no SDK retry, no repeat in another format, no other model or sign-in."""
    receipt, wire, _records = _probe_run(world, monkeypatch, provider, status=status)

    name, candidate = ("gateway", OPUS_NEW) if provider == "anthropic" else ("codex", SOL_NEW)
    assert [(http_sent["model"], _Http.credential(request))
            for http_sent, request in zip(wire.sent(), wire.requests)] == [(candidate, f"placeholder-{name}")]
    (checked,) = receipt["candidates"]
    assert [(probe["sign_in"], probe["passed"]) for probe in checked["probe"]] == [(name, False)]
    assert receipt["result"] == "nothing_filed" and _suggestions(world) == []


@pytest.mark.parametrize("provider", ["anthropic", "openai-codex"])
def test_probe_wire_carries_the_output_cap_and_a_two_minute_timeout(world, monkeypatch, provider):
    """Contract C: Anthropic sends max_tokens 8000; each provider's transport is bound to 120 s."""
    receipt, wire, _records = _probe_run(world, monkeypatch, provider)

    (request,) = wire.requests
    (sent,) = wire.sent()
    if provider == "anthropic":
        assert sent["max_tokens"] == 8000
    timeout = request.extensions["timeout"]
    assert set(timeout) == {"connect", "read", "write", "pool"}
    assert all(seconds is not None and 0 < seconds <= 120 for seconds in timeout.values()), timeout
    assert receipt["result"] == "filed"


@pytest.mark.parametrize("after,in_time", [(1.0, True), (121.0, False)], ids=["in-time", "late"])
@pytest.mark.parametrize("provider", ["anthropic", "openai-codex"])
def test_probe_reply_after_two_minutes_by_the_checks_clock_is_failed(world, monkeypatch, provider, after, in_time):
    """Contract C: the check's own clock decides; a late reply files nothing and keeps its real tokens."""
    receipt, wire, _records = _probe_run(world, monkeypatch, provider, after=after)

    name = "gateway" if provider == "anthropic" else "codex"
    assert len(wire.requests) == 1
    (checked,) = receipt["candidates"]
    assert _probes(checked) == [(name, in_time, 7, 1)]
    assert receipt["result"] == ("filed" if in_time else "nothing_filed")
    assert len(_suggestions(world)) == int(in_time)


@pytest.mark.parametrize("provider", ["anthropic", "openai-codex"])
def test_probe_deadline_runs_from_the_send_to_the_reply(world, monkeypatch, provider):
    """Contract C: a slow first client build (the SDK's import) is no late reply."""
    from agent import anthropic_adapter
    from agent import auxiliary_client as aux

    muc = _muc()
    env = _planner_pinned(world)
    wire = _Http(monkeypatch, after=2.0)
    owner, name = (anthropic_adapter, "build_anthropic_client") if provider == "anthropic" else (aux, "_create_openai_client")
    build = getattr(owner, name)

    def slow_build(*args, **kwargs):
        wire.now += 119.0
        return build(*args, **kwargs)

    monkeypatch.setattr(owner, name, slow_build)
    monkeypatch.setattr(muc, "_sign_in_value", lambda _provider, sign_in: f"placeholder-{sign_in}")
    receipt = _cli(world, replace(_probed(provider).seams(muc), probe=muc._probe_call), env=env)

    label = "gateway" if provider == "anthropic" else "codex"
    assert len(wire.requests) == 1
    (checked,) = receipt["candidates"]
    assert _probes(checked) == [(label, True, 7, 1)]
    assert [probe["seconds"] for probe in checked["probe"]] == [2.0]
    assert receipt["result"] == "filed"


@pytest.mark.parametrize("provider", ["anthropic", "openai-codex"])
def test_probe_refusal_echoing_the_sign_in_leaves_no_raw_log_record(world, monkeypatch, provider):
    """Contract D: no record from the client build to the reply read, nested errors included."""
    receipt, wire, records = _probe_run(world, monkeypatch, provider, value=LEAK, status=400)

    assert [_Http.credential(request) for request in wire.requests] == [LEAK]
    assert receipt["result"] == "nothing_filed" and _suggestions(world) == []
    (checked,) = receipt["candidates"]
    assert [probe["error"] for probe in checked["probe"]] == ["BadRequestError"]
    assert LEAK not in _raw_text(records)
    assert LEAK not in json.dumps(receipt) + _dump(world)


# ---------------------------------------------------------------------------
# The owner's two corrections of candidate 0d223b17: the account proof has one
# source, and one request reaches the transport. HTTPX's own send and redirect
# handling stay real here; only the transport below them is fake.
# ---------------------------------------------------------------------------


def _transport_wire(after=1.0):
    """``_Http``'s answers without its patch of ``httpx.Client.send``."""
    wire = object.__new__(_Http)
    wire.requests, wire.status, wire.after, wire.now = [], 200, after, 5000.0
    return wire


def _transport_run(world, monkeypatch, provider, wire, respond):
    """The probe through the real SDKs and HTTPX's own send; ``respond`` is the fake transport."""
    import httpx

    muc = _muc()
    transport = httpx.MockTransport(respond)
    monkeypatch.setattr(httpx.Client, "_transport_for_url", lambda _client, _url: transport)
    monkeypatch.setattr(muc, "_clock", lambda: wire.now)
    monkeypatch.setattr(muc, "_sign_in_value", lambda _provider, name: f"placeholder-{name}")
    return _cli(world, replace(_probed(provider).seams(muc), probe=muc._probe_call), env=_planner_pinned(world))


@pytest.mark.parametrize("provider", ["anthropic", "openai-codex"])
@pytest.mark.parametrize("answer", ["success", "refusal", "server-error", "redirect", "timeout"])
def test_cli_probe_sends_at_most_one_http_request(world, monkeypatch, provider, answer):
    """Contract B at the transport: one request whatever the provider answers, a redirect included."""
    import httpx

    wire = _transport_wire()

    def respond(request):
        first = not wire.requests
        # A repeat gets a valid answer, so a probe that repeated its request would pass and file.
        wire.status = {"refusal": 400, "server-error": 500}.get(answer, 200) if first else 200
        if first and answer == "redirect":
            wire.requests.append(request)
            location = str(request.url.copy_with(query=b"probe-hop=1"))
            return httpx.Response(307, headers={"location": location}, request=request)
        if first and answer == "timeout":
            wire.requests.append(request)
            raise httpx.ReadTimeout("the fake provider did not answer in time", request=request)
        return wire.answer(request)

    receipt = _transport_run(world, monkeypatch, provider, wire, respond)

    passed = answer == "success"
    name, candidate = ("gateway", OPUS_NEW) if provider == "anthropic" else ("codex", SOL_NEW)
    assert len(wire.requests) == 1, "one probe sent more than one HTTP request"
    assert [(sent["model"], _Http.credential(request))
            for sent, request in zip(wire.sent(), wire.requests)] == [(candidate, f"placeholder-{name}")]
    (checked,) = receipt["candidates"]
    assert [(probe["sign_in"], probe["passed"]) for probe in checked["probe"]] == [(name, passed)]
    assert receipt["result"] == ("filed" if passed else "nothing_filed")
    assert len(_suggestions(world)) == int(passed)


@pytest.mark.parametrize("provider", ["anthropic", "openai-codex"])
def test_cli_late_reply_fails_under_real_httpx_dispatch(world, monkeypatch, provider):
    """Contract C at the transport: a reply after two minutes by the check's clock files nothing."""
    wire = _transport_wire(after=121.0)

    receipt = _transport_run(world, monkeypatch, provider, wire, wire.answer)

    assert len(wire.requests) == 1
    assert receipt["result"] == "nothing_filed" and _suggestions(world) == []
    (checked,) = receipt["candidates"]
    (probe,) = checked["probe"]
    assert (probe["passed"], probe["late"], probe["input_tokens"], probe["output_tokens"]) == (False, True, 7, 1)


def _stale_login_row():
    """A naturally stale singleton-seeded row: its login file has since been removed."""
    return {"id": "placeholder-stale-row", "label": "placeholder stale login", "auth_type": "oauth",
            "source": "claude_code", "priority": 0, "access_token": "placeholder-stale-token",
            "base_url": "https://api.anthropic.com"}


def _in_profile_scope(home, reader):
    from agent.secret_scope import build_profile_secret_scope

    return _muc()._in_scope(home, build_profile_secret_scope(home), reader)


def test_cli_unprovable_legacy_fallback_does_not_rewrite_the_auth_store(world):
    """Contract A: no usable runtime choice is no proof, and no resolver behind the preview is asked."""
    from agent import credential_pool

    from hermes_cli import auth

    muc = _muc()
    env = _planner_pinned(world)
    home = world.root / "profiles" / PLANNER
    store = []
    _in_profile_scope(home, lambda: store.append(auth.write_credential_pool("anthropic", [_stale_login_row()])) or "")
    before = store[0].read_bytes()
    selected = []
    _in_profile_scope(home, lambda: selected.append(credential_pool.preview_runtime_selection("anthropic")[0]) or "")
    assert selected == [None], "the fake store must have no usable runtime choice"
    assert store[0].read_bytes() == before, "the preview itself is read-only"
    fake = Fake(routes=[(PLANNER, "anthropic", OPUS, "max", "routine")],
                lists={("anthropic", "gateway"): [OPUS, OPUS_NEW]})

    receipt = _cli(world, replace(fake.seams(muc), sign_ins=muc._role_sign_ins), env=env)

    assert store[0].read_bytes() == before, "a read-only update check rewrote the auth store"
    (entry,) = receipt["models"]
    assert (entry["result"], entry["role"]) == ("sign_in_unproven", PLANNER)
    assert receipt["result"] == "nothing_filed" and _suggestions(world) == []
    assert fake.probe_calls == []


def test_worker_sign_in_is_its_own_runtime_choice_and_no_store_is_rewritten(world, monkeypatch):
    """Contract A for the coding worker too: the preview's choice in its own scope, and no other source."""
    from hermes_cli import auth, models

    muc = _muc()
    planner, worker = world.root / "profiles" / PLANNER, world.root / "profiles" / WORKER
    worker.mkdir(parents=True, exist_ok=True)
    account = "placeholder-worker-account"
    row = {"id": "placeholder-worker-row", "label": "placeholder worker account", "auth_type": "api_key",
           "priority": 0, "source": "manual", "access_token": account, "base_url": "https://api.anthropic.com"}
    stores = []
    # The process's own store has no usable choice; the worker's own store has one account.
    _in_profile_scope(planner, lambda: stores.append(auth.write_credential_pool("anthropic", [_stale_login_row()])) or "")
    _in_profile_scope(worker, lambda: stores.append(auth.write_credential_pool("anthropic", [row])) or "")
    before = [store.read_bytes() for store in stores]
    asked = []
    monkeypatch.setattr(models, "_fetch_anthropic_models",
                        lambda *, api_key, **_kwargs: asked.append(api_key) or [OPUS, OPUS_NEW])
    fake = Fake(routes=[(WORKER, "anthropic", OPUS, "max", "deep")], lists={})

    receipt = _cli(world, replace(fake.seams(muc), sign_ins=muc._role_sign_ins, account_models=muc._account_model_ids))

    assert [store.read_bytes() for store in stores] == before, "a read-only update check rewrote an auth store"
    assert asked == [account]
    assert receipt["result"] == "filed" and len(_suggestions(world)) == 1
    assert account not in json.dumps(receipt) + _dump(world)


@pytest.mark.parametrize("build_state", ["different-unlisted", "different-listed", "same-account", "absent"])
def test_cli_worker_build_sign_in_must_list_the_candidate_and_gets_its_own_probe(world, monkeypatch, build_state):
    """The erratum: the coding worker's build sign-in, read only by ``read_build_token``, is a second account.

    Each distinct account must list the candidate and gets one probe; an equal
    value is the same account. No resolver is asked and nothing is rewritten.
    """
    from agent import anthropic_credentials
    from hermes_cli import auth, models, runtime_provider
    from plugins.dashboard_auth.raphael_workspace import build_sign_in_file, sandbox_dispatch

    muc = _muc()
    env = _planner_pinned(world)
    worker = world.root / "profiles" / WORKER
    worker.mkdir(parents=True, exist_ok=True)
    gateway = "placeholder-worker-gateway"
    build = gateway if build_state == "same-account" else "placeholder-worker-build"
    row = {"id": "placeholder-worker-row", "label": "placeholder worker account", "auth_type": "api_key",
           "priority": 0, "source": "manual", "access_token": gateway, "base_url": "https://api.anthropic.com"}
    stores = []
    _in_profile_scope(worker, lambda: stores.append(auth.write_credential_pool("anthropic", [row])) or "")
    if build_state == "absent":
        assert build_sign_in_file.read_build_token() is None
    else:
        build_sign_in_file.write_build_token(build)
        assert build_sign_in_file.read_build_token()[0] == build

    def held():
        # The worker's auth store and the root's build sign-in file; a file made by the run shows too.
        return {str(path): path.read_bytes()
                for path in [*stores, *(world.root / "secrets").glob("build-sign-in.json")]}

    before = held()
    resolved = []

    def resolver(*_args, **_kwargs):
        resolved.append(True)
        raise AssertionError("the update check called a resolver")

    # The adapter's resolve_anthropic_token is only a lazy pointer to this one.
    monkeypatch.setattr(sandbox_dispatch, "_resolve_host_credential", resolver)
    monkeypatch.setattr(anthropic_credentials, "resolve_anthropic_token", resolver)
    monkeypatch.setattr(runtime_provider, "_anthropic_env_runtime", resolver)
    listed = []

    def fetch(*, api_key, **_kwargs):
        listed.append(api_key)
        return [OPUS] if api_key == build and build_state == "different-unlisted" else [OPUS, OPUS_NEW]

    monkeypatch.setattr(models, "_fetch_anthropic_models", fetch)
    fake = Fake(routes=[(WORKER, "anthropic", OPUS, "max", "deep")], lists={})

    receipt = _cli(world, replace(fake.seams(muc), sign_ins=muc._role_sign_ins,
                                  account_models=muc._account_model_ids), env=env)

    probed = [call[2].value for call in fake.probe_calls]
    assert held() == before, "a read-only update check rewrote an auth store or the build sign-in file"
    assert resolved == [], "the update check called a resolver"
    for value in (gateway, build):
        assert value not in json.dumps(receipt) + _dump(world)
    if build_state == "different-unlisted":
        assert receipt["result"] == "nothing_filed", "an unlisted build account was ignored and a suggestion was filed"
        assert _suggestions(world) == [] and probed == []
        assert listed == [gateway, build]
        (checked,) = receipt["candidates"]
        assert (checked["result"], checked["account"]) == ("not_listed", {"gateway": True, "build": False})
        return
    assert receipt["result"] == "filed" and len(_suggestions(world)) == 1
    accounts = [gateway, build] if build_state == "different-listed" else [gateway]
    assert listed == accounts, "each distinct account is listed once"
    assert probed == accounts, "each distinct account is probed once"
    if build_state == "different-listed":
        # The receipt names the build sign-in by its name, never by its value.
        (entry,) = receipt["models"]
        assert entry["sign_ins"] == [{"sign_in": "gateway", "roles": [WORKER]},
                                     {"sign_in": "build", "roles": [WORKER]}]
        (checked,) = receipt["candidates"]
        assert [probe["sign_in"] for probe in checked["probe"]] == ["gateway", "build"]


# ---------------------------------------------------------------------------
# The catalog: a run judges a candidate only from a catalog refreshed in that
# run, once and before its first capability read. No test reaches the service.
# ---------------------------------------------------------------------------

CATALOG_NOT_REFRESHED = "The model catalog could not be refreshed."


def _catalog_copy(*models):
    """A copy of the model catalog with a capability row for each of ``models`` and no other."""
    row = {"reasoning": True, "tool_call": True, "limit": {"context": 200_000, "output": 64_000}}
    return {"anthropic": {"id": "anthropic", "models": {model: dict(row, id=model) for model in models}}}


def _no_held_catalog(monkeypatch, models_dev):
    """The process holds no copy of the catalog yet; the module's own values come back after the test."""
    for name, value in (("_models_dev_cache", {}), ("_models_dev_cache_time", 0), ("_models_dev_retry_after", 0)):
        monkeypatch.setattr(models_dev, name, value)


def _recorded(muc, fake, events, *, refreshed):
    """``fake``'s seams with a refresh that answers ``refreshed``; each refresh and row read goes to ``events``."""
    seams = fake.seams(muc)

    def refresh_catalog():
        events.append("refresh")
        return refreshed

    def capabilities(provider, model):
        events.append(("capabilities", model))
        return seams.capabilities(provider, model)

    return replace(seams, capabilities=capabilities, refresh_catalog=refresh_catalog)


def test_practice_run_judges_the_catalog_refreshed_in_the_run_not_its_old_copy(world, monkeypatch):
    from agent import models_dev

    muc = _muc()
    # The worker's own copy on disk is old: it has no row for the model the role uses now.
    old, refreshed = _catalog_copy("claude-opus-4-1"), _catalog_copy(OPUS)
    Path(world.env["HERMES_HOME"], "models_dev_cache.json").write_text(json.dumps(old), encoding="utf-8")
    _no_held_catalog(monkeypatch, models_dev)

    def refresh_catalog():
        # The service answered: the refreshed copy is the one the process holds.
        monkeypatch.setattr(models_dev, "_models_dev_cache", refreshed)
        return True

    fake = Fake(routes=[(PLANNER, "anthropic", OPUS, "max", "routine")], lists={("anthropic", "gateway"): [OPUS]})
    # A namespace, not a Seams: on the code before the refresh seam, this test still runs the check and fails.
    seams = SimpleNamespace(**{**vars(fake.seams(muc)), "capabilities": muc._capability_row,
                               "refresh_catalog": refresh_catalog})

    receipt = _command(world, seams, "--practice", PLANNER)

    (checked,) = receipt["candidates"]
    assert (checked["to_model"], checked["result"]) == (OPUS, "passed")
    assert (receipt["result"], receipt["practice"]) == ("filed", True)
    assert [row["recommendation_subject_id"] for row in _suggestions(world)] == [_update("anthropic", OPUS, OPUS)]


def test_failed_catalog_refresh_files_nothing_and_reads_no_capability_row_and_sends_no_probe(world):
    muc = _muc()
    fake = Fake(routes=[(PLANNER, "anthropic", OPUS, "max", "routine")],
                lists={("anthropic", "gateway"): [OPUS, OPUS_NEW]})
    events = []

    receipt = _command(world, _recorded(muc, fake, events, refreshed=False))

    assert _suggestions(world) == []
    assert (receipt["result"], receipt["reason"], receipt["filed"]) == ("nothing_filed", CATALOG_NOT_REFRESHED, None)
    # No candidate is judged from an old copy: no row is read and no probe is sent.
    assert events == ["refresh"]
    assert receipt["candidates"] == [] and fake.probe_calls == []
    # The receipt keeps its keys and its schema version.
    assert sorted(receipt) == ["boards_read", "budget", "candidates", "error", "filed", "models", "month",
                               "practice", "practice_role", "reason", "result", "schema_version", "unread_roles"]
    assert receipt["schema_version"] == 1
    # The board tool shows the reason, so it passes the owner text check and names no model, provider or address.
    reason = receipt["reason"]
    assert check_owner_text(reason, "planner_prose") == reason
    for name in (OPUS, OPUS_NEW, "anthropic", "openai", "codex", "claude", "opus", "gpt", "models.dev", "http"):
        assert name not in reason.lower()


def test_two_candidates_refresh_the_catalog_once_before_the_first_capability_read(world):
    muc = _muc()
    # The higher candidate has no row, so the lower one is judged too.
    fake = Fake(routes=[(PLANNER, "anthropic", OPUS, "max", "routine")],
                lists={("anthropic", "gateway"): [OPUS, OPUS_NEW, OPUS_NEWER]},
                capabilities={OPUS_NEWER: (None, None)})
    events = []

    receipt = _command(world, _recorded(muc, fake, events, refreshed=True))

    assert [checked["to_model"] for checked in receipt["candidates"]] == [OPUS_NEWER, OPUS_NEW]
    assert [checked["result"] for checked in receipt["candidates"]] == ["capability_unknown", "passed"]
    assert receipt["result"] == "filed"
    assert events == ["refresh", ("capabilities", OPUS_NEWER), ("capabilities", OPUS),
                      ("capabilities", OPUS_NEW), ("capabilities", OPUS)]


def test_run_with_no_candidate_makes_no_catalog_refresh(world):
    muc = _muc()
    fake = Fake(routes=[(PLANNER, "anthropic", OPUS, "max", "routine")], lists={("anthropic", "gateway"): [OPUS]})
    events = []

    receipt = _command(world, _recorded(muc, fake, events, refreshed=True))

    assert (receipt["result"], receipt["reason"]) == ("nothing_filed", "No newer version was found.")
    assert receipt["candidates"] == [] and events == []


@pytest.mark.parametrize(
    ("answer", "refreshed"),
    [(answer, answer != "no-answer") for answer in ("new-data", "not-modified", "no-answer")],
    ids=["new-data", "not-modified", "no-answer-old-copy-served"],
)
def test_default_catalog_refresh_is_the_catalogs_own_forced_fetch(tmp_path, monkeypatch, answer, refreshed):
    from agent import models_dev

    muc = _muc()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    old, new = _catalog_copy("claude-opus-4-1"), _catalog_copy(OPUS)
    models_dev._get_cache_path().write_text(json.dumps(old), encoding="utf-8")
    _no_held_catalog(monkeypatch, models_dev)

    def request(*, conditional=False):
        # The one request the fetch sends, standing in for the service.
        if answer == "new-data":
            return new, ""
        if answer == "not-modified":
            raise models_dev._NotModified()
        raise ConnectionError("the catalog service did not answer")

    forced_fetch, calls, served = models_dev.fetch_models_dev, [], []

    def fetch_models_dev(*args, **kwargs):
        calls.append((args, kwargs))
        served.append(forced_fetch(*args, **kwargs))
        return served[-1]

    monkeypatch.setattr(models_dev, "_fetch_models_dev_from_network", request)
    monkeypatch.setattr(models_dev, "fetch_models_dev", fetch_models_dev)

    assert muc.default_seams().refresh_catalog() is refreshed
    assert calls == [((), {"force_refresh": True})]
    # Not modified and no answer both serve the old copy: only the service's answer tells them apart.
    assert served == [new if answer == "new-data" else old]


def test_default_seams_refresh_is_the_real_forced_fetch_not_the_field_default(monkeypatch):
    from agent import models_dev

    muc = _muc()
    calls = []

    def fetch_models_dev(*args, **kwargs):
        # The stand-in for the forced fetch: the catalog service does not answer.
        calls.append((args, kwargs))
        raise ConnectionError("the catalog service did not answer")

    monkeypatch.setattr(models_dev, "fetch_models_dev", fetch_models_dev)

    # The field default would report a fresh catalog and make no call.
    assert muc.default_seams().refresh_catalog() is False
    assert calls == [((), {"force_refresh": True})]


def test_no_production_code_builds_seams_outside_default_seams():
    """Holds the owner's decision on the refresh seam's field default until that default is removed."""
    root = Path(__file__).resolve().parents[2]
    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    include = pyproject["tool"]["setuptools"]["packages"]["find"]["include"]
    # The production packages, then the root's single-file modules as setup.py derives them.
    files = [path for package in sorted({name.split(".")[0] for name in include})
             for path in sorted((root / package).rglob("*.py"))]
    files += [path for path in sorted(root.glob("*.py")) if path.name != "setup.py"]
    builders, importers = set(), set()
    for path in files:
        source = path.read_bytes()
        if b"Seams" not in source and b"refresh_catalog" not in source:
            continue
        name = path.relative_to(root).as_posix()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(source, filename=name)
        # Each call that builds Seams or passes a refresh seam, with its innermost enclosing function.
        nodes = [(tree, None)]
        while nodes:
            node, function = nodes.pop()
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                function = node.name
            if isinstance(node, ast.Call):
                callee = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", None)
                if callee == "Seams" or any(keyword.arg == "refresh_catalog" for keyword in node.keywords):
                    builders.add((name, function))
            if isinstance(node, ast.ImportFrom) and any(alias.name == "Seams" for alias in node.names):
                importers.add(name)
            nodes.extend((child, function) for child in ast.iter_child_nodes(node))

    assert builders == {("hermes_cli/model_update_check.py", "default_seams")}
    assert importers <= {"hermes_cli/model_update_check.py"}
