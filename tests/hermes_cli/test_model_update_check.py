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
import contextlib
import hashlib
import importlib
import json
import logging
import os
import shutil
import sqlite3
import subprocess
import threading
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path

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
