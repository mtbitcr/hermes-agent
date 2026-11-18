"""Undo the last model update and start a practice update, from Settings.

POST /api/model-updates/undo rolls the last applied model update back: its
record entry and every role it moved, in one guarded batch, audited per role.
POST /api/model-updates/practice files one card on the chosen Project's board
for the planning role, whose worker runs the practice check. Both serve the
Models machine only. These tests drive the real dashboard app with real machine
tokens over a real home: the update record, the role configs, the Projects and
their boards. Only the provider's model list and the check's outside reads are
stubbed.
"""

from __future__ import annotations

import builtins
import contextlib
import errno
import functools
import importlib
import importlib.util
import io
import itertools
import json
import os
import shutil
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
import yaml

from hermes_cli import config as hermes_config
from hermes_cli import kanban_db, owner_workspace as ow, projects_db
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import owner_model_updates as omu
from hermes_cli.dashboard_auth.audit import _resolve_log_path
from hermes_cli.profiles import get_profile_dir
from hermes_constants import get_default_hermes_root, get_hermes_home
from plugins.dashboard_auth.raphael_workspace import model_policy
from tests.gateway.test_api_server_owner_model_updates import (  # noqa: F401  (fixtures)
    _CURRENT as A,
    _NEWER as B,
    _PROVIDER,
    _SUBJECT,
    _events,
    _project,
    _record_graph_receipt,
    _suggestion,
    _write_route,
    live_models,
    owner,
    roles,
)
from tests.hermes_cli.test_model_update_check import Fake
from tests.hermes_cli.test_web_router_build_sign_in import (  # noqa: F401  (fixtures)
    GATED_REFUSALS,
    _audit_text,
    _holds_part_of,
    _refusals,
    dash,
    every_log_line,
)

ROUTER = "hermes_cli.web_routers.model_updates"
UNDO = "/api/model-updates/undo"
PRACTICE = "/api/model-updates/practice"
OAUTH = "/api/providers/oauth"
C = "claude-sonnet-5-7"
ROLE = "raphael-builder"
PLANNER = "raphael-planner"
MOVED = ["raphael-builder", "raphael-business"]  # the two roles on A
CHECKED = "model_update_checked"
NO_PROJECT = "That Project does not exist or has been archived."
SIGN_IN = "sk-ant-oat01-SYNTHETIC" + "Xq3Vm7Pz" * 6


@pytest.fixture()
def home():
    """The per-test root home the suite already isolates; the dashboard runs on it."""
    return get_hermes_home()


@pytest.fixture()
def team(dash, owner, roles, live_models):
    """The dashboard over two roles on A, the planner on another model, one Project."""
    project = _project(owner)
    return _Team(dash, owner, project, _slug(owner, project), live_models)


class _Team:
    def __init__(self, dash, owner, project, slug, listed):
        self.dash, self.call = dash, dash.call
        self.owner, self.project, self.slug, self.listed = owner, project, slug, listed


def _router():
    """The routes' module, asserted inside a test so its absence fails that test."""
    assert importlib.util.find_spec(ROUTER) is not None, f"{ROUTER} does not exist"
    return importlib.import_module(ROUTER)


def _slug(owner, project):
    [slug] = [p["slug"] for p in ow.list_committed_projects(owner) if p["project_id"] == project["project_id"]]
    return slug


def _undo(team, who="models", **kwargs):
    return team.call("POST", UNDO, who, **kwargs)


def _practice(team, role=ROLE, project=None, who="models", **kwargs):
    return team.call("POST", PRACTICE, who, json={"role": role, "project": project or team.slug}, **kwargs)


def _record_path():
    return get_default_hermes_root() / "raphael" / "model_policy_updates.json"


def _record_bytes():
    path = _record_path()
    return path.read_bytes() if path.exists() else None


def _entries():
    return [(e["from_model"], e["to_model"], e["state"]) for e in json.loads(_record_path().read_text())["updates"]]


def _configs():
    """Every admitted role's config.yaml, byte for byte; None for a role without one."""
    found = {}
    for profile in model_policy.admitted_profile_ids():
        path = get_profile_dir(profile) / "config.yaml"
        if profile != "default":
            found[profile] = path.read_bytes() if path.exists() else None
    return found


def _unchanged():
    """Everything a practice must not touch: every config, the record, the journal."""
    journal = hermes_config._route_batch_journal_path()
    return _configs(), _record_bytes(), journal.read_bytes() if journal.exists() else None


def _model(profile):
    return yaml.safe_load((get_profile_dir(profile) / "config.yaml").read_text())["model"]["default"]


def _revision(profile):
    return hermes_config.profile_route_revision(get_profile_dir(profile))


def _apply(team, subject=_SUBJECT):
    """Update on a filed suggestion through card 3's apply, as the owner's click does."""
    suggestion_id = _suggestion(team.project, subject)
    ref = ow._owner_decision_ref(team.owner, authority="recommendation", native_id=suggestion_id)
    omu.apply_model_update(team.owner, decision_ref=ref, reason="Update")
    return suggestion_id


def _undo_audit():
    from hermes_cli.dashboard_auth.audit import read_audit_records

    return [
        (record["profile"], record["model_provider"], record.get("model"))
        for record in read_audit_records()
        if record.get("source") == "raphael-models" and record.get("profile") in MOVED
    ]


def _board_file(board):
    return get_default_hermes_root() / "kanban" / "boards" / board / "kanban.db"


def _rows(path, sql, args=()):
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(row) for row in conn.execute(sql, args)]
    finally:
        conn.close()


def _tasks(team):
    return _rows(_board_file(team.project["board"]), "SELECT * FROM tasks WHERE task_kind = 'work' ORDER BY rowid")


def _practice_cards(team, role=ROLE):
    key = f"model-update-practice:{role}:{time.strftime('%Y-%m', time.gmtime())}"
    return [task for task in _tasks(team) if task["idempotency_key"] == key]


def _suggestion_counts():
    """Model update suggestions per board file, counted straight off disk."""
    root = get_default_hermes_root()
    files = {"default": root / "kanban.db"}
    files.update({path.parent.name: path for path in (root / "kanban" / "boards").glob("*/kanban.db")})
    return {
        board: _rows(path, "SELECT COUNT(*) AS n FROM tasks WHERE task_kind = 'recommendation' "
                           "AND recommendation_subject_id LIKE 'model-update:%'")[0]["n"]
        for board, path in files.items() if path.exists()
    }


def _running_work(team, role="raphael-business"):
    """Receipt-owned owner work of ``role``, unsealed and running now: the fence refuses."""
    with contextlib.closing(kanban_db.connect(board=team.project["board"])) as conn:
        task_id = kanban_db.create_task(
            conn, title="work in progress", assignee=role, execution_tier="routine",
            project_id=team.project["project_id"],
        )
        conn.execute("UPDATE tasks SET status = 'running' WHERE id = ?", (task_id,))
        conn.commit()
    _record_graph_receipt(team.owner, team.project, [task_id])
    return task_id


def _finish(team, task_id):
    with contextlib.closing(kanban_db.connect(board=team.project["board"])) as conn:
        conn.execute("UPDATE tasks SET status = 'done' WHERE id = ?", (task_id,))
        conn.commit()


def _sealed(team, model, title):
    lock = kanban_db.mint_policy_lock(ROLE, _PROVIDER, model, "max", "routine")
    with contextlib.closing(kanban_db.connect(board=team.project["board"])) as conn:
        return kanban_db.create_task(
            conn, title=title, assignee=ROLE, execution_tier="routine",
            project_id=team.project["project_id"], model_override=model,
            provider_override=_PROVIDER, reasoning_effort="max", model_policy_lock=lock,
        )


def _seal_errors(team, ids):
    """``(task_policy_lock_error, policy_lock_error)`` for each card's stored seal."""
    found = []
    for task in _tasks(team):
        if task["id"] in ids:
            found.append((kanban_db.task_policy_lock_error(task), model_policy.policy_lock_error(
                task["model_policy_lock"], task["assignee"], task["provider_override"],
                task["model_override"], task["reasoning_effort"], task["execution_tier"],
            )))
    assert len(found) == len(ids)
    return found


def _gate(monkeypatch, owner, name):
    """The first two calls of ``owner.name`` wait for each other, then run as they would."""
    real = getattr(owner, name)
    barrier = threading.Barrier(2, timeout=20)
    calls = itertools.count()

    @functools.wraps(real)
    def gated(*args, **kwargs):
        if next(calls) < 2:
            barrier.wait()
        return real(*args, **kwargs)

    monkeypatch.setattr(owner, name, gated)


def _together(*calls):
    results, errors = [None] * len(calls), []

    def run(index):
        try:
            results[index] = calls[index]()
        except BaseException as exc:  # surfaced below, never swallowed
            errors.append(exc)

    threads = [threading.Thread(target=run, args=(index,)) for index in range(len(calls))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(60)
    assert not errors, errors
    return results


def _worker_env(team, card, board=None):
    """The env the dispatcher gives ``card``'s worker: claimed and spawned for real."""
    board = board or team.project["board"]
    with contextlib.closing(kanban_db.connect(board=board)) as conn:
        task = kanban_db.claim_task(conn, card)
    assert task is not None and task.current_run_id is not None
    root = get_default_hermes_root()
    workspace = root / "kanban" / "boards" / board / "workspaces" / card
    workspace.mkdir(parents=True)
    captured = {}

    class _Proc:
        pid = 4242

    def _popen(cmd, *args, **kwargs):
        captured["env"] = dict(kwargs["env"])
        return _Proc()

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(subprocess, "Popen", _popen)
        kbd._default_spawn(task, str(workspace), board=board)
    env = captured["env"]
    assert env["HERMES_KANBAN_DB"] == str(_board_file(board))
    assert env["HERMES_KANBAN_BOARD"] == board
    assert env["HERMES_KANBAN_TASK"] == card
    assert env["HERMES_HOME"] == str(root / "profiles" / PLANNER)
    return env


def _run_check(env, fake, *, practice_role=None):
    """Run the check card's command as the worker: the dispatcher's env and nothing else."""
    muc = importlib.import_module("hermes_cli.model_update_check")
    with pytest.MonkeyPatch.context() as patch:
        for key in [key for key in os.environ if key not in env]:
            patch.delenv(key)
        for key, value in env.items():
            if os.environ.get(key) != value:
                patch.setenv(key, value)
        return muc.run_update_check(practice_role=practice_role, seams=fake.seams(muc))


def _receipts_on(board, card):
    return _rows(_board_file(board), "SELECT payload FROM task_events WHERE task_id = ? AND kind = ?",
                 (card, CHECKED))


def _model_update_items(team):
    return [item for item in ow.list_owner_decisions(team.owner, model_update=True)["data"]
            if item.get("kind") == "model_update"]


# ---------------------------------------------------------------------------
# Undo
# ---------------------------------------------------------------------------


def test_u1_undo_moves_both_roles_back_and_records_audits_and_receipts_it(team):
    suggestion_id = _apply(team)
    prior = {role: _revision(role) for role in MOVED}
    assert [_model(role) for role in MOVED] == [B, B]

    resp = _undo(team)

    assert resp.status_code == 200, resp.text
    assert resp.json() == {"state": "undone", "roles": MOVED, "from_model": A, "to_model": B}
    assert _entries() == [(A, B, "rolled_back")]
    assert [_model(role) for role in MOVED] == [A, A]
    revisions = {role: _revision(role) for role in MOVED}
    assert all(revisions[role] != prior[role] for role in MOVED)
    [event] = _events(team.project, suggestion_id, "model_update_rolled_back")
    assert (event["from_model"], event["to_model"], event["roles"]) == (A, B, MOVED)
    assert event["revisions"] == revisions and event["prior_revisions"] == prior
    assert sorted(_undo_audit()) == [(role, _PROVIDER, A) for role in MOVED]


def test_u2_seals_on_both_versions_hold_through_a_failed_and_a_done_undo(team):
    before = _sealed(team, A, "sealed before the update")
    _apply(team)
    after = _sealed(team, B, "sealed after the update")
    clean = [(None, None), (None, None)]
    assert _seal_errors(team, {before, after}) == clean

    running = _running_work(team)
    assert _undo(team).json()["state"] == "busy"
    assert _seal_errors(team, {before, after}) == clean

    _finish(team, running)
    assert _undo(team).json()["state"] == "undone"
    assert _seal_errors(team, {before, after}) == clean
    assert kanban_db.mint_policy_lock(ROLE, _PROVIDER, A, "max", "routine")
    with pytest.raises(ValueError):
        kanban_db.mint_policy_lock(ROLE, _PROVIDER, B, "max", "routine")


def test_u3_no_record_answers_none_and_writes_nothing(team):
    configs = _configs()
    assert _record_bytes() is None

    resp = _undo(team)

    assert resp.status_code == 200, resp.text
    assert resp.json() == {"state": "none"}
    assert not _record_path().exists()
    assert _configs() == configs


def test_u4_a_second_undo_answers_already_undone_and_writes_nothing(team):
    _apply(team)
    assert _undo(team).json()["state"] == "undone"
    record, configs = _record_bytes(), _configs()

    resp = _undo(team)

    assert resp.status_code == 200, resp.text
    assert resp.json() == {"state": "already_undone", "from_model": A, "to_model": B}
    assert _record_bytes() == record
    assert _configs() == configs


def test_u5_running_unsealed_work_answers_busy_and_changes_nothing(team):
    _apply(team)
    _running_work(team)
    record, configs = _record_bytes(), _configs()

    resp = _undo(team)

    assert resp.status_code == 200, resp.text
    assert resp.json() == {"state": "busy", "roles": ["raphael-business"]}
    assert _record_bytes() == record
    assert _configs() == configs


def test_busy_names_every_role_whose_work_is_running(team):
    """Section 5.9: busy lists the roles whose work runs, not only the first refused."""
    _apply(team)
    for role in MOVED:
        _running_work(team, role=role)
    record, configs = _record_bytes(), _configs()

    assert _undo(team).json() == {"state": "busy", "roles": MOVED}
    assert (_record_bytes(), _configs()) == (record, configs)


def test_u6_a_role_moved_after_the_record_was_read_is_not_applied(team, monkeypatch):
    _apply(team)
    record, configs = _record_bytes(), _configs()
    flip = model_policy.apply_model_update_record

    def moved_first(update):
        # Settings moves one role after the undo read the roles and revisions.
        _write_route("raphael-business", "openai-codex", "gpt-5.6-terra", "max")
        configs.update(_configs())
        return flip(update)

    monkeypatch.setattr(model_policy, "apply_model_update_record", moved_first)
    resp = _undo(team)

    assert resp.status_code == 200, resp.text
    assert resp.json() == {"state": "not_applied"}
    assert _record_bytes() == record
    assert _configs() == configs


def test_u7_two_undos_at_once_give_one_undone_and_write_each_config_once(team, monkeypatch):
    _apply(team)
    writes = []
    save = hermes_config.save_config

    def counted(*args, **kwargs):
        writes.append(get_hermes_home().name)
        return save(*args, **kwargs)

    monkeypatch.setattr(hermes_config, "save_config", counted)
    _gate(monkeypatch, hermes_config, "_route_batch_journal_lock")

    answers = _together(lambda: _undo(team), lambda: _undo(team))

    assert [resp.status_code for resp in answers] == [200, 200]
    assert sorted(resp.json()["state"] for resp in answers) == ["already_undone", "undone"]
    assert sorted(writes) == MOVED
    assert _entries() == [(A, B, "rolled_back")]


def test_u8_undo_of_a_chained_update_moves_back_one_step(team):
    team.listed[_PROVIDER] = [A, B, C, "claude-opus-5-5"]
    _apply(team)
    _apply(team, f"model-update:{_PROVIDER}:{B}:{C}:2026-11")
    assert [_model(role) for role in MOVED] == [C, C]

    resp = _undo(team)

    assert resp.json() == {"state": "undone", "roles": MOVED, "from_model": B, "to_model": C}
    assert [_model(role) for role in MOVED] == [B, B]
    assert _entries() == [(A, B, "active"), (B, C, "rolled_back")]
    assert model_policy.configured_assignment_for(ROLE).model == B


def test_u9_an_unreadable_record_answers_503_and_writes_nothing(team):
    _apply(team)
    _record_path().write_text("{not a record", encoding="utf-8")
    record, configs = _record_bytes(), _configs()
    with pytest.raises(model_policy.EnrollmentUnavailable):
        model_policy.read_model_updates()

    resp = _undo(team)

    assert resp.status_code == 503
    assert resp.json() == {"detail": "Service Unavailable"}
    assert _record_bytes() == record
    assert _configs() == configs


# ---------------------------------------------------------------------------
# Undo recovery: an interrupted undo, the commit marker and the receipt
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _restore_route_latch():
    """A test that latches the route journal does not latch the next one."""
    before = hermes_config._ROUTE_JOURNAL_BLOCKED
    try:
        yield
    finally:
        hermes_config._ROUTE_JOURNAL_BLOCKED = before


class _AbruptStop(BaseException):
    """A stop that no ``except Exception`` sees, as a process end is."""


def _undo_state_path():
    return get_default_hermes_root() / "raphael" / "model_policy_updates.undo.json"


def _rolled_back(team, suggestion_id):
    return _events(team.project, suggestion_id, "model_update_rolled_back")


def _finished(team, suggestion_id, prior):
    """The undo is whole: record rolled back, both roles on A, one receipt, no pending state."""
    assert _entries() == [(A, B, "rolled_back")]
    assert [_model(role) for role in MOVED] == [A, A]
    [event] = _rolled_back(team, suggestion_id)
    assert event["revisions"] == {role: _revision(role) for role in MOVED}
    assert (event["roles"], event["prior_revisions"]) == (MOVED, prior)
    assert not _undo_state_path().exists()


def test_an_audit_that_cannot_begin_restores_everything(team, monkeypatch):
    suggestion_id = _apply(team)
    before = (_record_bytes(), _configs())
    prior = {role: _revision(role) for role in MOVED}

    def fail(*args, **kwargs):
        raise model_policy.AuditWriteError("synthetic pre-commit failure")

    with monkeypatch.context() as patch:
        patch.setattr(model_policy, "begin_audit_batch", fail)
        resp = _undo(team)

    assert (resp.status_code, resp.json()) == (503, {"detail": "Service Unavailable"})
    assert (_record_bytes(), _configs()) == before
    assert _undo_audit() == []
    assert _undo_state_path().exists()
    assert _undo(team).json() == {"state": "undone", "roles": MOVED, "from_model": A, "to_model": B}
    _finished(team, suggestion_id, prior)
    assert sorted(_undo_audit()) == [(role, _PROVIDER, A) for role in MOVED]


def test_a_commit_marker_that_cannot_be_written_leaves_no_successful_audit(team, monkeypatch):
    suggestion_id = _apply(team)
    before = (_record_bytes(), _configs())
    prior = {role: _revision(role) for role in MOVED}
    original = hermes_config._write_route_batch_journal_record

    def fail_marker(record):
        if record["phase"] == hermes_config._ROUTE_BATCH_AUDIT_COMMITTED:
            raise OSError(errno.EIO, "synthetic journal marker I/O failure")
        return original(record)

    with monkeypatch.context() as patch:
        patch.setattr(hermes_config, "_write_route_batch_journal_record", fail_marker)
        resp = _undo(team)

    assert (resp.status_code, resp.json()) == (503, {"detail": "Service Unavailable"})
    assert (_record_bytes(), _configs()) == before
    assert _undo_audit() == []
    assert _rolled_back(team, suggestion_id) == []
    assert hermes_config.reconcile_profile_route_batch_journal() == []  # the next start
    assert _undo(team).json()["state"] == "undone"
    _finished(team, suggestion_id, prior)
    assert sorted(_undo_audit()) == [(role, _PROVIDER, A) for role in MOVED]


def test_an_audit_that_cannot_commit_after_the_marker_answers_503_and_the_undo_stands(team, monkeypatch):
    suggestion_id = _apply(team)
    prior = {role: _revision(role) for role in MOVED}

    def fail(*args, **kwargs):
        raise model_policy.AuditWriteError("synthetic commit failure")

    with monkeypatch.context() as patch:
        patch.setattr(model_policy, "commit_audit_batch", fail)
        resp = _undo(team)

    assert (resp.status_code, resp.json()) == (503, {"detail": "Service Unavailable"})
    assert _entries() == [(A, B, "rolled_back")]
    assert [_model(role) for role in MOVED] == [A, A]
    assert _undo_audit() == []
    assert _undo(team).json()["state"] == "undone"
    assert sorted(_undo_audit()) == [(role, _PROVIDER, A) for role in MOVED]
    assert _undo(team).json()["state"] == "already_undone"
    _finished(team, suggestion_id, prior)
    assert sorted(_undo_audit()) == [(role, _PROVIDER, A) for role in MOVED]


def test_an_unreconciled_journal_refuses_the_undo_before_it_changes_anything(team, monkeypatch):
    def fail_retirement():
        raise hermes_config._block_route_journal("synthetic retirement I/O failure")

    with monkeypatch.context() as patch:
        patch.setattr(hermes_config, "_clear_route_batch_journal", fail_retirement)
        _apply(team)
    assert hermes_config._ROUTE_JOURNAL_BLOCKED is not None
    before = (_record_bytes(), _configs())

    resp = _undo(team)

    assert (resp.status_code, resp.json()) == (503, {"detail": "Service Unavailable"})
    assert (_record_bytes(), _configs()) == before
    assert hermes_config._ROUTE_JOURNAL_BLOCKED is not None
    assert not _undo_state_path().exists()


def test_a_receipt_that_cannot_be_written_answers_503_and_the_retry_writes_it_once(team, monkeypatch):
    suggestion_id = _apply(team)
    prior = {role: _revision(role) for role in MOVED}

    def locked(*args, **kwargs):
        raise sqlite3.OperationalError("database is locked")

    with monkeypatch.context() as patch:
        patch.setattr(omu, "_append_receipt", locked)
        first = _undo(team)

    assert (first.status_code, first.json()) == (503, {"detail": "Service Unavailable"})
    assert _rolled_back(team, suggestion_id) == []
    assert _undo_state_path().exists()
    for state in ("undone", "already_undone"):
        resp = _undo(team)
        assert (resp.status_code, resp.json()["state"]) == (200, state)
    _finished(team, suggestion_id, prior)
    assert sorted(_undo_audit()) == [(role, _PROVIDER, A) for role in MOVED]


def test_an_undo_whose_suggestion_is_gone_stands_without_a_receipt(team, monkeypatch):
    suggestion_id = _apply(team)
    monkeypatch.setattr(ow, "_owner_decision_ref", lambda *args, **kwargs: "decision:gone")

    resp = _undo(team)

    assert resp.json() == {"state": "undone", "roles": MOVED, "from_model": A, "to_model": B}
    assert _rolled_back(team, suggestion_id) == []
    assert not _undo_state_path().exists()


def test_a_stop_right_after_the_receipt_does_not_write_it_twice(team, monkeypatch):
    suggestion_id = _apply(team)
    prior = {role: _revision(role) for role in MOVED}
    original = omu._append_receipt

    def stop_after(*args, **kwargs):
        original(*args, **kwargs)
        raise _AbruptStop()

    with monkeypatch.context() as patch:
        patch.setattr(omu, "_append_receipt", stop_after)
        with pytest.raises(BaseException):
            _undo(team)

    assert _undo(team).json()["state"] == "already_undone"
    _finished(team, suggestion_id, prior)


def test_a_stop_after_the_first_config_is_settled_by_the_next_start_and_undo(team, monkeypatch):
    suggestion_id = _apply(team)
    before = (_record_bytes(), _configs())
    prior = {role: _revision(role) for role in MOVED}
    original = omu._write_model

    def stop_after_first_config(*args, **kwargs):
        original(*args, **kwargs)
        raise _AbruptStop()

    with monkeypatch.context() as patch:
        patch.setattr(omu, "_write_model", stop_after_first_config)
        with pytest.raises(BaseException):
            _undo(team)

    assert hermes_config.reconcile_profile_route_batch_journal() == MOVED  # the next start
    assert _configs() == before[1]
    resp = _undo(team)
    assert resp.json() == {"state": "undone", "roles": MOVED, "from_model": A, "to_model": B}
    _finished(team, suggestion_id, prior)
    assert sorted(_undo_audit()) == [(role, _PROVIDER, A) for role in MOVED]


def test_no_update_is_recorded_while_an_interrupted_undo_is_unsettled(team, monkeypatch):
    _apply(team)
    original = omu._write_model

    def stop_after_first_config(*args, **kwargs):
        original(*args, **kwargs)
        raise _AbruptStop()

    with monkeypatch.context() as patch:
        patch.setattr(omu, "_write_model", stop_after_first_config)
        with pytest.raises(BaseException):
            _undo(team)
    hermes_config.reconcile_profile_route_batch_journal()
    pending = (_record_bytes(), _configs())
    [entry] = json.loads(_record_path().read_text())["updates"]
    later = {**entry, "decision_ref": entry["decision_ref"] + ":later", "state": "active"}

    with pytest.raises(model_policy.EnrollmentUnavailable):
        model_policy.apply_model_update_record(later)

    assert (_record_bytes(), _configs()) == pending
    assert _undo(team).json()["state"] == "undone"
    assert _entries() == [(A, B, "rolled_back")]
    assert [_model(role) for role in MOVED] == [A, A]
    model_policy.apply_model_update_record(later)
    assert _entries() == [(A, B, "rolled_back"), (A, B, "active")]


def _exit_after(module, name):
    original = getattr(module, name)

    def exit_after(*args, **kwargs):
        original(*args, **kwargs)
        os._exit(77)

    setattr(module, name, exit_after)


def _exit_before(module, name):
    setattr(module, name, lambda *args, **kwargs: os._exit(77))


_STOPS = {
    "after the record flip": lambda: _exit_after(model_policy, "apply_model_update_record"),
    "after the first config": lambda: _exit_after(omu, "_write_model"),
    "after every config, before the commit": lambda: _exit_before(hermes_config, "mark_route_batch_audit_committed"),
    "after the commit marker": lambda: _exit_after(hermes_config, "mark_route_batch_audit_committed"),
}


@pytest.mark.linux_only
@pytest.mark.parametrize("stop", sorted(_STOPS))
def test_a_real_process_exit_then_the_cli_start_and_the_next_undo_leave_the_undo_whole(team, stop):
    suggestion_id = _apply(team)
    before = (_record_bytes(), _configs())
    prior = {role: _revision(role) for role in MOVED}
    child = os.fork()
    if child == 0:
        signal.alarm(30)
        _STOPS[stop]()
        _undo(team)
        os._exit(98)
    _, status = os.waitpid(child, 0)
    assert os.waitstatus_to_exitcode(status) == 77
    restart = os.fork()
    if restart == 0:
        signal.alarm(30)
        from hermes_cli import main

        importlib.reload(main)
        os._exit(0)
    _, status = os.waitpid(restart, 0)
    assert os.waitstatus_to_exitcode(status) == 0
    assert not hermes_config._route_batch_journal_path().exists()
    committed = stop == "after the commit marker"
    assert (_configs() == before[1]) is not committed

    resp = _undo(team)

    assert resp.status_code == 200, resp.text
    assert resp.json()["state"] == "undone"
    _finished(team, suggestion_id, prior)
    assert sorted(_undo_audit()) == [(role, _PROVIDER, A) for role in MOVED]


# ---------------------------------------------------------------------------
# Owner rule 1: each seam of the undo, each read of its role set, two faults
# ---------------------------------------------------------------------------

# Each call of the undo that writes, in its order: the seam, then the module
# attribute its fault wraps and which call of it. The restore writes run only
# after a fault in the batch, so they are reached through _BATCH_FAULT first.
_SEAMS = [
    ("01 undo state, prepared", omu, "_save_undo_state", 1),
    ("02 record flip", model_policy, "apply_model_update_record", 1),
    ("03 route write, first role", omu, "_write_model", 1),
    ("04 route write, second role", omu, "_write_model", 2),
    ("05 enrollment mark", hermes_config, "mark_route_batch_enrollment_written", 1),
    ("06 audit begin", model_policy, "begin_audit_batch", 1),
    ("07 batch commit mark", hermes_config, "mark_route_batch_audit_committed", 1),
    ("08 undo state, committed", omu, "_save_undo_state", 2),
    ("09 audit commit", model_policy, "commit_audit_batch", 1),
    ("10 event", omu, "_append_receipt", 1),
    ("11 undo state removal", omu, "_clear_undo_state", 1),
    ("12 restore of a config", hermes_config, "_restore_config_snapshot", 1),
    ("13 restore of the record", model_policy, "restore_model_update_record", 1),
]
_RESTORES = {"12 restore of a config", "13 restore of the record"}
_BATCH_FAULT = (omu, "_write_model", 2)


def _fault(patch, module, name, call, kind):
    """At the ``call``-th call of ``module.name``: an error in its place, or a real exit right after it."""
    original = getattr(module, name)
    calls = itertools.count(1)

    def faulted(*args, **kwargs):
        if next(calls) != call:
            return original(*args, **kwargs)
        if kind == "error":
            raise OSError(errno.EIO, f"synthetic fault at {name}")
        original(*args, **kwargs)
        os._exit(77)

    patch.setattr(module, name, faulted)


def _seam_faults(patch, seam, kind):
    name, module, attribute, call = seam
    if name in _RESTORES:
        _fault(patch, *_BATCH_FAULT, "error")
    _fault(patch, module, attribute, call, kind)


def _stores(team, suggestion_id):
    """All the undo writes: the record, each config, its audit and its event."""
    return _record_bytes(), _configs(), sorted(_undo_audit()), _rolled_back(team, suggestion_id)


def _whole(team, suggestion_id, before):
    """The one whole outcome the stores hold, A or B, with no undo state; fails on any other."""
    assert not _undo_state_path().exists()
    if _entries() == [(A, B, "rolled_back")]:
        assert [_model(role) for role in MOVED] == [A, A]
        assert sorted(_undo_audit()) == [(role, _PROVIDER, A) for role in MOVED]
        assert len(_rolled_back(team, suggestion_id)) == 1
        return "A"
    assert (_record_bytes(), _configs()) == before
    assert (_undo_audit(), _rolled_back(team, suggestion_id)) == ([], [])
    return "B"


def _answer_is_true(resp, team, suggestion_id, before, earlier):
    """Point 2 of the rule; ``earlier`` is what the stores held before this call."""
    if resp.status_code == 503:
        assert resp.json() == {"detail": "Service Unavailable"}
        assert _undo_state_path().exists() or _stores(team, suggestion_id) == earlier
        return
    assert resp.status_code == 200, resp.text
    state = resp.json()["state"]
    assert state in ("undone", "already_undone", "not_applied", "busy")
    if state == "already_undone":
        assert _stores(team, suggestion_id) == earlier
    assert _whole(team, suggestion_id, before) == ("A" if state.endswith("undone") else "B")


@pytest.mark.parametrize("seam", _SEAMS, ids=[seam[0] for seam in _SEAMS])
def test_an_error_at_each_seam_then_one_more_undo_ends_whole(team, monkeypatch, seam):
    suggestion_id = _apply(team)
    before = (_record_bytes(), _configs())
    earlier = _stores(team, suggestion_id)
    with monkeypatch.context() as patch:
        _seam_faults(patch, seam, "error")
        first = _undo(team)
    _answer_is_true(first, team, suggestion_id, before, earlier)

    earlier = _stores(team, suggestion_id)
    resp = _undo(team)

    assert resp.status_code == 200, resp.text
    _answer_is_true(resp, team, suggestion_id, before, earlier)


@pytest.mark.linux_only
@pytest.mark.parametrize("seam", _SEAMS, ids=[seam[0] for seam in _SEAMS])
def test_a_real_exit_right_after_each_seam_then_one_more_undo_ends_whole(team, seam):
    suggestion_id = _apply(team)
    before = (_record_bytes(), _configs())
    child = os.fork()
    if child == 0:
        try:
            signal.alarm(30)
            with pytest.MonkeyPatch.context() as patch:
                _seam_faults(patch, seam, "exit")
                _undo(team)
        finally:
            os._exit(98)
    _, status = os.waitpid(child, 0)
    assert os.waitstatus_to_exitcode(status) == 77
    earlier = _stores(team, suggestion_id)

    resp = _undo(team)

    assert resp.status_code == 200, resp.text
    _answer_is_true(resp, team, suggestion_id, before, earlier)


def _read_fails(name):
    """A predicate that is true for the call ``name`` of the role-set read makes for the first moved role."""
    home = get_profile_dir(MOVED[0])
    return {
        "admitted roles": lambda *args: sys._getframe(2).f_globals.get("__name__") == omu.__name__,
        "configured route": lambda *args: args[:1] == (MOVED[0],),
        "native config read": lambda *args: get_hermes_home() == home,
        "route revision": lambda *args: args[:1] == (home,),
    }[name]


_ROLE_READS = {
    "admitted roles": (model_policy, "admitted_profile_ids"),
    "configured route": (model_policy, "configured_assignment_for"),
    "native config read": (hermes_config, "load_config_readonly"),
    "route revision": (hermes_config, "profile_route_revision"),
}


@pytest.mark.parametrize("read", sorted(_ROLE_READS))
def test_a_role_set_read_that_fails_answers_503_and_changes_nothing(team, monkeypatch, read):
    _apply(team)
    before = (_record_bytes(), _configs())
    module, attribute = _ROLE_READS[read]
    original, fails = getattr(module, attribute), _read_fails(read)

    def failing(*args, **kwargs):
        if fails(*args):
            raise PermissionError(errno.EACCES, "synthetic role read failure")
        return original(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(module, attribute, failing)
        resp = _undo(team)

    assert (resp.status_code, resp.json()) == (503, {"detail": "Service Unavailable"})
    assert (_record_bytes(), _configs()) == before
    assert _undo_audit() == []
    assert not _undo_state_path().exists()


def test_a_batch_fault_and_a_restore_fault_answer_503_keep_the_state_and_the_next_undo_ends_whole(
    team, monkeypatch,
):
    suggestion_id = _apply(team)
    before = (_record_bytes(), _configs())
    with monkeypatch.context() as patch:
        _fault(patch, *_BATCH_FAULT, "error")
        patch.setattr(hermes_config, "_restore_config_snapshot", lambda *args, **kwargs: False)
        first = _undo(team)

    assert (first.status_code, first.json()) == (503, {"detail": "Service Unavailable"})
    assert _undo_state_path().exists()
    earlier = _stores(team, suggestion_id)
    resp = _undo(team)
    assert resp.json() == {"state": "undone", "roles": MOVED, "from_model": A, "to_model": B}
    _answer_is_true(resp, team, suggestion_id, before, earlier)


def _pending_undo(team, patch):
    """An undo whose event cannot be written while ``patch`` lasts: 503, and its state stays."""
    suggestion_id = _apply(team)

    def locked(*args, **kwargs):
        raise sqlite3.OperationalError("database is locked")

    patch.setattr(omu, "_append_receipt", locked)
    assert _undo(team).status_code == 503
    assert _undo_state_path().exists()
    return suggestion_id


def test_an_update_while_an_undo_is_pending_answers_503_then_settles_first(team, monkeypatch):
    from gateway.platforms import api_server_owner_model_updates as update_route

    later = _suggestion(team.project, f"model-update:{_PROVIDER}:{A}:{B}:2026-10")
    ref = ow._owner_decision_ref(team.owner, authority="recommendation", native_id=later)
    with monkeypatch.context() as patch:
        suggestion_id = _pending_undo(team, patch)
        pending = (_record_bytes(), _configs(), _undo_state_path().read_bytes())
        with pytest.raises(ow.OwnerWorkspaceError) as refused:
            omu.apply_model_update(team.owner, decision_ref=ref, reason="Update")
        assert refused.value.code not in update_route._REFUSALS  # the Update route answers 503
        assert (_record_bytes(), _configs(), _undo_state_path().read_bytes()) == pending
        assert omu._pending_update(team.owner, ref)[1] == later  # still waiting

    omu.apply_model_update(team.owner, decision_ref=ref, reason="Update")

    assert len(_rolled_back(team, suggestion_id)) == 1
    assert not _undo_state_path().exists()
    assert _entries() == [(A, B, "rolled_back"), (A, B, "active")]
    assert [_model(role) for role in MOVED] == [B, B]
    assert sorted(_undo_audit()) == [(role, _PROVIDER, A) for role in MOVED]


def test_a_practice_while_an_undo_is_pending_answers_503_then_settles_first(team, monkeypatch):
    with monkeypatch.context() as patch:
        suggestion_id = _pending_undo(team, patch)
        pending = (_unchanged(), _undo_state_path().read_bytes(), _tasks(team))
        resp = _practice(team)
        assert (resp.status_code, resp.json()) == (503, {"detail": "Service Unavailable"})
        assert (_unchanged(), _undo_state_path().read_bytes(), _tasks(team)) == pending

    resp = _practice(team)

    assert resp.json() == {"state": "started", "roles": [ROLE]}
    assert len(_practice_cards(team)) == 1
    assert len(_rolled_back(team, suggestion_id)) == 1
    assert not _undo_state_path().exists()
    assert _entries() == [(A, B, "rolled_back")]
    assert [_model(role) for role in MOVED] == [A, A]


def test_a_pending_undo_state_holds_no_sign_in_value(team, monkeypatch, tmp_path):
    # The Models machine's own secret, from the token file the dash fixture minted.
    secret = (tmp_path / "machine-tokens" / "models.token").read_text(encoding="utf-8").strip().rsplit(".", 1)[1]
    monkeypatch.setenv("ANTHROPIC_API_KEY", SIGN_IN)
    with monkeypatch.context() as patch:
        _pending_undo(team, patch)
        text = _undo_state_path().read_text(encoding="utf-8")

    assert json.loads(text)["audit"]
    for value in (SIGN_IN, secret):
        assert not _holds_part_of(text, value)


# ---------------------------------------------------------------------------
# Practice
# ---------------------------------------------------------------------------


def test_p1_practice_files_one_sealed_planning_card_and_changes_nothing(team):
    month = time.strftime("%Y-%m", time.gmtime())
    before, tasks = _unchanged(), _tasks(team)

    resp = _practice(team)

    assert resp.status_code == 200, resp.text
    assert resp.json() == {"state": "started", "roles": [ROLE]}
    [card] = [task for task in _tasks(team) if task not in tasks]
    assert (card["assignee"], card["execution_tier"]) == (PLANNER, "routine")
    assert card["project_id"] == team.project["project_id"]
    assert card["idempotency_key"] == f"model-update-practice:{ROLE}:{month}"
    assert f"hermes model update-check --practice {ROLE}" in card["body"]
    assert card["model_policy_lock"]
    assert _seal_errors(team, {card["id"]}) == [(None, None)]
    assert _unchanged() == before


def test_p2_a_second_click_in_the_same_month_answers_already_started(team):
    assert _practice(team).json()["state"] == "started"

    resp = _practice(team)

    assert resp.status_code == 200, resp.text
    assert resp.json() == {"state": "already_started", "roles": [ROLE]}
    assert len(_practice_cards(team)) == 1


def test_p3_two_clicks_at_once_file_one_card(team, monkeypatch):
    _gate(monkeypatch, ow, "list_committed_projects")

    answers = _together(lambda: _practice(team), lambda: _practice(team))

    assert [resp.status_code for resp in answers] == [200, 200]
    assert sorted(resp.json()["state"] for resp in answers) == ["already_started", "started"]
    assert len(_practice_cards(team)) == 1


def test_p4_an_unknown_role_or_project_or_an_archived_project_is_refused(team):
    garden = _project(team.owner, "Garden")
    garden_slug = _slug(team.owner, garden)
    with projects_db.connect_closing() as conn:
        assert projects_db.archive_project(conn, garden["project_id"])
    tasks = _tasks(team)

    refused = {
        "unknown role": _practice(team, role="raphael-stranger"),
        "a role that is not text": _practice(team, role=7),
        "unknown project": _practice(team, project="no-such-project"),
        "archived project": _practice(team, project=garden_slug),
    }

    assert {label: (resp.status_code, resp.json()) for label, resp in refused.items()} == {
        "unknown role": (400, {"detail": "Invalid role"}),
        "a role that is not text": (400, {"detail": "Invalid role"}),
        "unknown project": (400, {"detail": NO_PROJECT}),
        "archived project": (400, {"detail": NO_PROJECT}),
    }
    assert _tasks(team) == tasks
    assert _rows(_board_file(garden["board"]), "SELECT id FROM tasks WHERE idempotency_key LIKE "
                                                "'model-update-practice:%'") == []


def test_p5_the_practice_card_s_worker_files_one_practice_suggestion_on_that_board(team):
    assert _practice(team).json()["state"] == "started"
    [card] = _practice_cards(team)
    env = _worker_env(team, card["id"])
    before, counts = _unchanged(), _suggestion_counts()
    fake = Fake(routes=[(ROLE, _PROVIDER, A, "max", "routine")], lists={(_PROVIDER, "gateway"): [A, B]})

    receipt = _run_check(env, fake, practice_role=ROLE)

    after = _suggestion_counts()
    board = team.project["board"]
    assert {slug: after[slug] - counts.get(slug, 0) for slug in after} == {
        slug: int(slug == board) for slug in after
    }
    assert (receipt["result"], receipt["practice"], receipt["filed"]["board"]) == ("filed", True, board)
    [row] = _rows(_board_file(board), "SELECT * FROM tasks WHERE task_kind = 'recommendation'")
    assert row["recommendation_subject_id"] == f"model-update:{_PROVIDER}:{A}:{A}:2026-11"
    [item] = _model_update_items(team)
    assert item["practice"] is True
    assert len(_receipts_on(board, card["id"])) == 1
    assert _unchanged() == before


def test_a_failed_practice_run_files_no_decision_and_its_receipt_is_on_the_card(team):
    """Owner decision 2: a failed practice puts nothing on Decisions; the card keeps why."""
    assert _practice(team).json()["state"] == "started"
    [card] = _practice_cards(team)
    env = _worker_env(team, card["id"])
    counts = _suggestion_counts()
    fake = Fake(
        routes=[(ROLE, _PROVIDER, A, "max", "routine")],
        lists={(_PROVIDER, "gateway"): [A, B]},
        answers={A: RuntimeError("the sample piece of work failed")},
    )

    receipt = _run_check(env, fake, practice_role=ROLE)

    assert receipt["result"] != "filed" and not receipt.get("filed")
    assert _suggestion_counts() == counts
    assert _model_update_items(team) == []
    [stored] = _receipts_on(team.project["board"], card["id"])
    assert json.loads(stored["payload"])["result"] == receipt["result"]


def test_after_an_undo_the_check_does_not_suggest_the_undone_pair_again(team):
    """Owner decision 3, from a worker pinned to another Project's board."""
    _apply(team)
    assert _undo(team).json()["state"] == "undone"
    garden = _project(team.owner, "Garden")
    with contextlib.closing(kanban_db.connect(board=garden["board"])) as conn:
        card = kanban_db.create_task(conn, title="Monthly model check", assignee=PLANNER,
                                     project_id=garden["project_id"])
    env = _worker_env(team, card, board=garden["board"])
    counts = _suggestion_counts()
    assert counts[team.project["board"]] == 1 and counts[garden["board"]] == 0
    fake = Fake(
        routes=[(role, _PROVIDER, _model(role), "max", "routine") for role in (*MOVED, PLANNER)],
        lists={(_PROVIDER, "gateway"): [A, B, "claude-opus-5-5"]},
    )

    receipt = _run_check(env, fake)

    assert _suggestion_counts() == counts
    assert not receipt.get("filed")
    assert team.project["board"] in receipt["boards_read"]
    assert [(c["to_model"], c["result"]) for c in receipt["candidates"]] == [(B, "accepted_before")]
    assert len(_receipts_on(garden["board"], card)) == 1


# ---------------------------------------------------------------------------
# Both routes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", [UNDO, PRACTICE])
def test_r1_a_models_token_route_with_strict_audit_and_404_to_anyone_else(team, path):
    from hermes_cli.dashboard_auth.token_auth import get_token_route_policy
    from plugins.dashboard_auth.raphael_workspace import token_store

    _router()
    policy = get_token_route_policy("POST", path)
    assert policy is not None
    assert (policy.required_scope, policy.optional, policy.strict_audit) == (token_store.MODELS_SCOPE, True, True)
    _apply(team)
    record, configs, tasks = _record_bytes(), _configs(), _tasks(team)

    resp = team.call("POST", path, "owner", json={"role": ROLE, "project": team.slug})

    assert (resp.status_code, resp.json()) == (404, {"detail": "Not found"})
    assert (_record_bytes(), _configs(), _tasks(team)) == (record, configs, tasks)


@pytest.mark.parametrize("path", [UNDO, PRACTICE])
def test_r2_another_profile_or_another_token_is_refused_as_for_the_build_sign_in(team, path):
    _apply(team)
    record, configs, tasks = _record_bytes(), _configs(), _tasks(team)
    body = {"json": {"role": ROLE, "project": team.slug}}

    def same_refusal(label, who, params, extra, via=None):
        oauth = team.call("GET", OAUTH, who, params=params, extra=extra, via=via)
        assert oauth.status_code >= 400, (label, oauth.status_code)
        resp = team.call("POST", path, who, params=params, extra=extra, via=via, **body)
        assert resp.status_code == oauth.status_code, (label, resp.status_code, resp.text)

    for label, who, params, extra in _refusals(team.dash):
        same_refusal(label, who, params, extra)
    with team.dash.gated() as gated_client:
        for label, who, extra in GATED_REFUSALS:
            same_refusal(label, who, None, extra, via=gated_client)
    other = team.call("POST", path, "models", params={"profile": "raphael-planner"}, **body)

    assert (other.status_code, other.json()) == (400, {"detail": "Invalid profile"})
    assert (_record_bytes(), _configs(), _tasks(team)) == (record, configs, tasks)


def test_r3_no_answer_log_record_or_audit_entry_holds_a_sign_in_value(team, every_log_line, monkeypatch):
    answers = []

    def call(status, path, **kwargs):
        resp = team.call("POST", path, "models", **kwargs)
        assert resp.status_code == status, (path, resp.text)
        answers.append(resp.text)

    call(200, UNDO)
    _apply(team)
    monkeypatch.setenv("ANTHROPIC_API_KEY", SIGN_IN)
    call(200, UNDO)
    call(200, UNDO)
    for _ in range(2):
        call(200, PRACTICE, json={"role": ROLE, "project": team.slug})
    call(400, PRACTICE, json={"role": SIGN_IN, "project": team.slug})
    call(400, PRACTICE, json={"role": ROLE, "project": SIGN_IN})

    def failing(error):
        def fail(*_args, **_kwargs):
            raise error
        return fail

    # Failures whose own error text carries the sign-in value.
    monkeypatch.setattr(model_policy, "read_model_updates", failing(model_policy.EnrollmentUnavailable(SIGN_IN)))
    call(503, UNDO)
    monkeypatch.setattr(kanban_db, "create_task", failing(RuntimeError(SIGN_IN)))
    call(503, PRACTICE, json={"role": "raphael-business", "project": team.slug})

    assert every_log_line.records
    audit = _audit_text()
    assert audit
    assert not _holds_part_of(audit, SIGN_IN)
    for _, _, text in every_log_line.records:
        assert not _holds_part_of(text, SIGN_IN), text
    for text in answers:
        assert not _holds_part_of(text, SIGN_IN), text


def test_every_new_owner_text_passes_the_owner_text_check_and_names_no_model():
    from hermes_cli.owner_text import check_owner_text

    texts = {
        "undone": "The last model update was undone. New work uses the earlier version again.",
        "none": "There is no model update to undo.",
        "already_undone": "The last model update was already undone.",
        "busy": "Some work is running right now, so nothing changed. Try again when it has finished.",
        "not_applied": "Raphael could not undo this update, so nothing changed.",
        "started": "A practice update is on its way. It will show in Decisions in a few minutes.",
        "already_started": "A practice update for this part of your team has already started this month.",
    }
    assert omu.UNDO_TEXTS == {key: texts[key] for key in ("undone", "none", "already_undone", "busy", "not_applied")}
    assert omu.PRACTICE_TEXTS == {key: texts[key] for key in ("started", "already_started")}
    assert check_owner_text(omu.PRACTICE_CARD_TITLE, "title") == omu.PRACTICE_CARD_TITLE
    for text in (*texts.values(), omu.PRACTICE_CARD_TITLE):
        assert check_owner_text(text, "stop_reason_decision") == text
        for name in ("claude", "sonnet", "opus", "gpt", "anthropic", "openai"):
            assert name not in text.lower()


# ---------------------------------------------------------------------------
# Owner rule 2: the fault sweep by call count, the deciding proof
# ---------------------------------------------------------------------------

_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC


class _Calls:
    """Point 6: count each call of one run inside the test's Hermes home.

    Counted: each open of a file to read or write (as builtins and pathlib
    open it, and ``os.open``), each rename, removal and sync, and each open
    and commit of a board database. The audit store is a file, so its open
    and its commit (the sync) count the same way. At call ``fault`` an
    ``"error"`` raises a real ``OSError(EIO)`` instead of the call; an
    ``"exit"`` ends the process right after the ``fault``-th call that writes.
    """

    def __init__(self, fault=0, mode="error"):
        self.fault, self.mode, self.calls, self.paths = fault, mode, [], []
        self.root = os.path.realpath(get_default_hermes_root())
        self._lock, self._fds = threading.Lock(), {}

    def _inside(self, target):
        try:
            if isinstance(target, int) or hasattr(target, "fileno"):
                fd = target if isinstance(target, int) else target.fileno()
                try:
                    target = os.readlink(f"/proc/self/fd/{fd}")
                except OSError:
                    target = self._fds.get(fd, "")
            target = os.fsdecode(os.fspath(target))
            if target.startswith("file:"):
                target = target[5:].split("?", 1)[0]
            real = os.path.realpath(target)
        except (OSError, TypeError, ValueError):
            return False
        return real == self.root or real.startswith(self.root + os.sep)

    def call(self, kind, target, writes, run):
        if not self._inside(target):
            return run()
        with self._lock:
            self.calls.append((kind, writes))
            self.paths.append(os.path.relpath(os.fsdecode(os.fspath(target)), self.root)
                              if isinstance(target, (str, bytes, os.PathLike)) else repr(target))
            k = len(self.calls)
            w = len(self.writes)
        if k == self.fault and self.mode == "error":
            raise OSError(errno.EIO, f"synthetic input-output error at call {k} ({kind})")
        result = run()
        if writes and w == self.fault and self.mode == "exit":
            os._exit(77)
        return result

    def _counted_connection(self, database):
        calls = self

        class Counted(sqlite3.Connection):
            def execute(self, sql, *args, **kwargs):
                run = functools.partial(sqlite3.Connection.execute, self, sql, *args, **kwargs)
                if str(sql).lstrip().upper().startswith(("COMMIT", "END")):
                    return calls.call("commit", database, True, run)
                return run()

            def commit(self):
                return calls.call("commit", database, True, functools.partial(sqlite3.Connection.commit, self))

        return Counted

    def __enter__(self):
        real_open, real_os_open, real_connect = io.open, os.open, sqlite3.connect
        originals = {name: getattr(os, name) for name in ("replace", "rename", "remove", "unlink", "fsync")}

        def counted_open(file, mode="r", *args, **kwargs):
            handle = self.call("open", file, any(c in mode for c in "wax+"),
                               lambda: real_open(file, mode, *args, **kwargs))
            if isinstance(file, (str, bytes, os.PathLike)):
                with contextlib.suppress(Exception):
                    self._fds[handle.fileno()] = os.fspath(file)
            return handle

        def counted_os_open(path, flags, *args, **kwargs):
            fd = self.call("os.open", path, bool(flags & _WRITE_FLAGS),
                           lambda: real_os_open(path, flags, *args, **kwargs))
            self._fds[fd] = os.fspath(path)
            return fd

        def counted(name, writes=True):
            def run(target, *args, **kwargs):
                return self.call(name, target, writes, lambda: originals[name](target, *args, **kwargs))
            return run

        def counted_connect(database, *args, **kwargs):
            if "factory" not in kwargs and not args and self._inside(database):
                kwargs["factory"] = self._counted_connection(database)
            return self.call("connect", database, False, lambda: real_connect(database, *args, **kwargs))

        self._patch = pytest.MonkeyPatch()
        self._patch.setattr(builtins, "open", counted_open)
        self._patch.setattr(io, "open", counted_open)
        self._patch.setattr(os, "open", counted_os_open)
        for name in originals:
            self._patch.setattr(os, name, counted(name))
        self._patch.setattr(sqlite3, "connect", counted_connect)
        return self

    def __exit__(self, *exc):
        self._patch.undo()
        return False

    @property
    def writes(self):
        return [k for k, (_, writes) in enumerate(self.calls, start=1) if writes]


def _eio(*args, **kwargs):
    raise OSError(errno.EIO, "synthetic input-output error")


def _stop_after_first(original):
    def stop_after(*args, **kwargs):
        original(*args, **kwargs)
        raise _AbruptStop()

    return stop_after


# Point 7: the fault-free undo, and the settle (by the next undo) of each
# state a fault can leave.
_SWEEP = {
    "fault-free undo of two roles": None,
    "settle after a fault in the event write": lambda patch: patch.setattr(omu, "_append_receipt", _eio),
    "settle after a fault in the audit commit": lambda patch: patch.setattr(model_policy, "commit_audit_batch", _eio),
    "settle after a fault between the two route writes":
        lambda patch: patch.setattr(omu, "_write_model", _stop_after_first(omu._write_model)),
}


def _sweep_start(team, scenario, saved):
    """Build the scenario's start state once and keep a copy of the home."""
    suggestion_id = _apply(team)
    before = (_record_bytes(), _configs())
    if _SWEEP[scenario] is not None:
        with pytest.MonkeyPatch.context() as patch:
            _SWEEP[scenario](patch)
            try:
                assert _undo(team).status_code == 503
            except BaseException as exc:  # the stop, as the test client's task group raises it
                if not isinstance(exc, (_AbruptStop, BaseExceptionGroup)):
                    raise
        assert _undo_state_path().exists()
    shutil.copytree(get_default_hermes_root(), saved, symlinks=True)
    return suggestion_id, before


def _restore_home(saved):
    """The scenario's start again: the home's bytes, and no process memory of a run."""
    root = get_default_hermes_root()
    shutil.rmtree(root)
    shutil.copytree(saved, root, symlinks=True)
    hermes_config._ROUTE_JOURNAL_BLOCKED = None
    hermes_config._LOAD_CONFIG_CACHE.clear()
    hermes_config._LAST_EXPANDED_CONFIG_BY_PATH.clear()


def _counted_undo(team, fault=0):
    """One undo through the signed-in route with an EIO at call ``fault``."""
    with _Calls(fault) as calls:
        try:
            resp = _undo(team)
        except Exception as exc:  # what escaped the app, as the test client raises it
            resp = exc
    return resp, calls


def _fault_free_count(team, saved, suggestion_id, before):
    """Point 7: N from a fault-free run, the same on a repeat run."""
    runs = []
    for _ in range(2):
        _restore_home(saved)
        earlier = _stores(team, suggestion_id)
        resp, calls = _counted_undo(team)
        _answer_is_true(resp, team, suggestion_id, before, earlier)
        assert resp.status_code == 200 and _whole(team, suggestion_id, before) == "A"
        runs.append(calls)
    assert len(runs[0].calls) > 0
    assert runs[0].calls == runs[1].calls, "the fault-free count differs on a repeat run"
    return runs[0]


def _first_then_second(team, suggestion_id, before, earlier, first, where):
    """Point 9: a true first answer, then one fault-free undo that ends whole."""
    assert not isinstance(first, BaseException), f"{where}: the first answer is an exception: {first!r}"
    if first.status_code == 503:  # the route's 503, or the sign-in's own before the route
        assert first.json()["detail"] == "Service Unavailable", f"{where}: {first.text}"
        if not _undo_state_path().exists():
            _whole(team, suggestion_id, before)
    else:
        _answer_is_true(first, team, suggestion_id, before, earlier)
    earlier = _stores(team, suggestion_id)
    second = _undo(team)
    assert second.status_code == 200, f"{where}: second answer {second.status_code} {second.text}"
    _answer_is_true(second, team, suggestion_id, before, earlier)
    return _whole(team, suggestion_id, before)


@pytest.mark.parametrize("scenario", sorted(_SWEEP))
def test_rule2_an_error_at_each_counted_call_then_one_more_undo_ends_whole(team, tmp_path, scenario):
    suggestion_id, before = _sweep_start(team, scenario, tmp_path / "saved-home")
    n = len(_fault_free_count(team, tmp_path / "saved-home", suggestion_id, before).calls)
    print(f"RULE2_SWEEP error {scenario}: N={n}")
    for k in range(1, n + 1):
        _restore_home(tmp_path / "saved-home")
        earlier = _stores(team, suggestion_id)
        first, calls = _counted_undo(team, fault=k)
        assert len(calls.calls) >= k, f"call {k} was not reached"
        _first_then_second(team, suggestion_id, before, earlier, first, f"call {k} {calls.calls[k - 1]} {calls.paths[k - 1]}")


@pytest.mark.linux_only
@pytest.mark.parametrize("scenario", sorted(_SWEEP))
def test_rule2_a_real_exit_right_after_each_counted_write_then_one_more_undo_ends_whole(team, tmp_path, scenario):
    suggestion_id, before = _sweep_start(team, scenario, tmp_path / "saved-home")
    writes = _fault_free_count(team, tmp_path / "saved-home", suggestion_id, before).writes
    print(f"RULE2_SWEEP exit {scenario}: writes={len(writes)}")
    assert writes
    seen = tmp_path / "child-calls.json"
    for k in range(1, len(writes) + 1):
        _restore_home(tmp_path / "saved-home")
        child = os.fork()
        if child == 0:
            try:
                signal.alarm(60)
                with _Calls(k, "exit") as calls:
                    _undo(team)
                seen.write_text(json.dumps(calls.calls))
            finally:
                os._exit(98)
        _, status = os.waitpid(child, 0)
        assert os.waitstatus_to_exitcode(status) == 77, (
            f"write {k}: the child did not stop there; it made {seen.read_text() if seen.exists() else '?'}")
        hermes_config._ROUTE_JOURNAL_BLOCKED = None
        hermes_config._LOAD_CONFIG_CACHE.clear()
        hermes_config._LAST_EXPANDED_CONFIG_BY_PATH.clear()
        if not _undo_state_path().exists():
            _whole(team, suggestion_id, before)
        earlier = _stores(team, suggestion_id)
        second = _undo(team)
        assert second.status_code == 200, f"write {k}: second answer {second.status_code} {second.text}"
        _answer_is_true(second, team, suggestion_id, before, earlier)
        _whole(team, suggestion_id, before)


# ---------------------------------------------------------------------------
# The probes of review rounds one to three (attachments 1830, 1841, 1849),
# each with its fault and its assertions; the autouse latch fixture above
# restores the route journal latch after each.
# ---------------------------------------------------------------------------


@pytest.mark.linux_only
def test_review_r1_real_process_exit_and_cli_startup_keep_record_and_configs_together(team):
    _apply(team)
    before = (_record_bytes(), _configs())
    child = os.fork()
    if child == 0:
        signal.alarm(30)
        original = omu._write_model

        def exit_after_durable_config(*args, **kwargs):
            original(*args, **kwargs)
            os._exit(77)

        omu._write_model = exit_after_durable_config
        _undo(team)
        os._exit(98)
    _, status = os.waitpid(child, 0)
    assert os.waitstatus_to_exitcode(status) == 77
    assert hermes_config._route_batch_journal_path().exists()
    restart = os.fork()
    if restart == 0:
        signal.alarm(30)
        from hermes_cli import main

        importlib.reload(main)
        os._exit(0)
    _, status = os.waitpid(restart, 0)
    assert os.waitstatus_to_exitcode(status) == 0
    assert not hermes_config._route_batch_journal_path().exists()
    _undo(team)
    assert _record_bytes() == before[0] or all(_model(role) == A for role in MOVED)


def test_review_r1_audit_write_failure_before_commit_restores_everything(team, monkeypatch):
    _apply(team)
    before = (_record_bytes(), _configs())

    def fail(*args, **kwargs):
        raise model_policy.AuditWriteError("synthetic pre-commit failure")

    monkeypatch.setattr(model_policy, "begin_audit_batch", fail)
    resp = _undo(team)
    assert resp.status_code == 503
    assert (_record_bytes(), _configs()) == before
    assert _undo_audit() == []


def test_review_r1_a_committed_audit_never_describes_reverted_routes(team, monkeypatch):
    _apply(team)
    original = hermes_config._write_route_batch_journal_record

    def fail_marker(record):
        if record["phase"] == hermes_config._ROUTE_BATCH_AUDIT_COMMITTED:
            raise OSError(errno.EIO, "synthetic journal marker I/O failure")
        return original(record)

    monkeypatch.setattr(hermes_config, "_write_route_batch_journal_record", fail_marker)
    _undo(team)
    actual = {role: _model(role) for role in MOVED}
    assert all(actual[role] == model for role, provider, model in _undo_audit())


def test_review_r1_unreconciled_journal_refuses_before_flipping_record(team, monkeypatch):
    def fail_retirement():
        raise hermes_config._block_route_journal("synthetic retirement I/O failure")

    with monkeypatch.context() as patch:
        patch.setattr(hermes_config, "_clear_route_batch_journal", fail_retirement)
        _apply(team)
    assert hermes_config._ROUTE_JOURNAL_BLOCKED is not None
    before = (_record_bytes(), _configs())
    resp = _undo(team)
    assert resp.status_code == 503
    assert (_record_bytes(), _configs()) == before


def test_review_r1_receipt_failure_is_not_a_permanent_unrecorded_success(team, monkeypatch):
    suggestion_id = _apply(team)

    def unavailable(*args, **kwargs):
        raise sqlite3.OperationalError("database is locked")

    with monkeypatch.context() as patch:
        patch.setattr(omu, "_append_receipt", unavailable)
        first = _undo(team)
    _undo(team)
    assert _events(team.project, suggestion_id, "model_update_rolled_back") or first.status_code >= 500


def test_review_r1_restart_reconciles_record_with_interrupted_configs(team, monkeypatch):
    _apply(team)
    before = (_record_bytes(), _configs())
    with monkeypatch.context() as patch:
        patch.setattr(omu, "_write_model", _stop_after_first(omu._write_model))
        try:
            _undo(team)
        except BaseException:
            pass
        else:
            pytest.fail("expected the simulated shutdown to interrupt the served route")
    hermes_config.reconcile_profile_route_batch_journal()
    _undo(team)
    assert _record_bytes() == before[0] or all(_model(role) == A for role in MOVED)


def test_review_r2_audit_commit_failure_then_retry_keeps_a_durable_per_role_audit(team, monkeypatch):
    _apply(team)

    def fail(*args, **kwargs):
        raise model_policy.AuditWriteError("synthetic audit commit storage failure")

    with monkeypatch.context() as patch:
        patch.setattr(model_policy, "commit_audit_batch", fail)
        _undo(team)
    second = _undo(team)
    assert second.status_code >= 500 or sorted(_undo_audit()) == [(r, _PROVIDER, A) for r in MOVED]


@pytest.mark.linux_only
def test_review_r2_crash_after_marker_cannot_erase_the_per_role_audit(team):
    _apply(team)
    child = os.fork()
    if child == 0:
        signal.alarm(30)
        original = hermes_config.mark_route_batch_audit_committed

        def stop(*args, **kwargs):
            original(*args, **kwargs)
            os._exit(77)

        hermes_config.mark_route_batch_audit_committed = stop
        _undo(team)
        os._exit(98)
    _, status = os.waitpid(child, 0)
    assert os.waitstatus_to_exitcode(status) == 77
    restart = os.fork()
    if restart == 0:
        signal.alarm(30)
        from hermes_cli import main

        importlib.reload(main)
        os._exit(0)
    _, status = os.waitpid(restart, 0)
    assert os.waitstatus_to_exitcode(status) == 0
    retry = _undo(team)
    assert retry.status_code >= 500 or sorted(_undo_audit()) == [(r, _PROVIDER, A) for r in MOVED]


def test_review_r2_failed_config_rollback_is_not_acknowledged_as_nothing_changed(team, monkeypatch):
    _apply(team)
    before = (_record_bytes(), _configs())
    original = omu._write_model

    def write_then_error(*args, **kwargs):
        if args[1] == MOVED[1]:
            raise OSError(errno.EIO, "synthetic second-profile config device I/O failure")
        return original(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(omu, "_write_model", write_then_error)
        patch.setattr(hermes_config, "_restore_config_snapshot", lambda *args, **kwargs: False)
        first = _undo(team)
    assert (_record_bytes(), _configs()) == before or first.status_code >= 500


def test_review_r2_unreadable_role_route_is_unknown_not_a_successful_undo(team, monkeypatch):
    _apply(team)
    original = model_policy.configured_assignment_for

    def unreadable(profile, *args, **kwargs):
        if profile == MOVED[0]:
            raise model_policy.EnrollmentUnavailable("synthetic role config could not be read")
        return original(profile, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(model_policy, "configured_assignment_for", unreadable)
        result = _undo(team)
    assert result.status_code >= 500 or all(_model(r) == A for r in MOVED)


def test_review_r2_native_config_read_io_failure_does_not_omit_an_affected_role(team, monkeypatch):
    _apply(team)
    original = hermes_config.load_config_readonly
    target = get_profile_dir(MOVED[0])

    def fail_native_read(*args, **kwargs):
        if get_hermes_home() == target:
            raise PermissionError(errno.EACCES, "synthetic unreadable config.yaml")
        return original(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(hermes_config, "load_config_readonly", fail_native_read)
        result = _undo(team)
    assert result.status_code >= 500 or all(_model(r) == A for r in MOVED)


def test_review_r3_explicit_authority_store_read_error_is_not_an_absent_role(team, monkeypatch):
    _apply(team)
    before = (_record_bytes(), _configs())
    target, original_text = _record_path(), Path.read_text
    original_assignment = model_policy.configured_assignment_for
    scope, errors = threading.local(), []

    def in_role_scope(profile, *args, **kwargs):
        scope.target = profile == MOVED[0]
        try:
            return original_assignment(profile, *args, **kwargs)
        finally:
            scope.target = False

    def native_read_error(path, *args, **kwargs):
        if path == target and getattr(scope, "target", False):
            errors.append(str(path))
            raise OSError(errno.EIO, "synthetic model policy record I/O outage during role authority read")
        return original_text(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(model_policy, "configured_assignment_for", in_role_scope)
        patch.setattr(Path, "read_text", native_read_error)
        answer = _undo(team)
    assert errors, "the native storage error was not reached"
    assert (answer.status_code, answer.json()) == (503, {"detail": "Service Unavailable"})
    assert (_record_bytes(), _configs()) == before
    assert not _undo_state_path().exists()


def _r3_pending_after_audit(team, monkeypatch):
    suggestion_id = _apply(team)
    with monkeypatch.context() as patch:
        patch.setattr(omu, "_append_receipt", _eio)
        assert _undo(team).status_code == 503
    assert sorted(_undo_audit()) == [(r, _PROVIDER, A) for r in MOVED]
    return suggestion_id


def _r3_one_audit_read_error(monkeypatch, call):
    target, original, errors = _resolve_log_path(), Path.read_text, []

    def unreadable_once(path, *args, **kwargs):
        if path == target and not errors:
            errors.append(str(path))
            raise OSError(errno.EIO, "synthetic audit log read outage")
        return original(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "read_text", unreadable_once)
        answer = call()
    return answer, len(errors)


def test_review_r3_native_audit_read_error_cannot_duplicate_existing_records(team, monkeypatch):
    _r3_pending_after_audit(team, monkeypatch)
    _r3_one_audit_read_error(monkeypatch, lambda: _undo(team))
    assert sorted(_undo_audit()) == [(r, _PROVIDER, A) for r in MOVED]


def test_review_r3_control_no_prior_committed_audit_has_no_duplicate(team, monkeypatch):
    _apply(team)
    _r3_one_audit_read_error(monkeypatch, lambda: _undo(team))
    assert sorted(_undo_audit()) == [(r, _PROVIDER, A) for r in MOVED]


def test_review_r3_control_pending_audit_and_no_read_fault_settles_once(team, monkeypatch):
    suggestion_id = _r3_pending_after_audit(team, monkeypatch)
    answer = _undo(team)
    assert answer.status_code == 200
    assert sorted(_undo_audit()) == [(r, _PROVIDER, A) for r in MOVED]
    assert len(_events(team.project, suggestion_id, "model_update_rolled_back")) == 1
    assert _undo(team).json()["state"] == "already_undone"
