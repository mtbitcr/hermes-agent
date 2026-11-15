"""The designer is admitted to the Raphael build machine tools, read-only.

``raphael-designer`` renders its own option pages into pictures inside its own
Server 2 machine, the way ``raphael-verifier`` and ``raphael-planner`` read code
in theirs: the exact tracked source, no coding credential, no credential vault
or proxy, no authority to hand a patch back to the host. Proven on the real
tool path against the fake Server 2 and host fixture of
test_raphael_sandbox_dispatch: the seat is admitted only on its own card and
run, the kernel still refuses a designer patch, the artifact tool is no wider
for it than for the other read-only roles, and every role admitted before keeps
exactly the authority it had.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.delegation_context import delegated_child_context
from hermes_cli import kanban_db as kb
from plugins.dashboard_auth.raphael_workspace import sandbox_artifacts as sa
from plugins.dashboard_auth.raphael_workspace import sandbox_dispatch as sd
from plugins.dashboard_auth.raphael_workspace.model_policy import task_assignment_for
from tests.plugins.dashboard_auth.test_raphael_sandbox_dispatch import (  # noqa: F401,E501
    FakeSandbox, REAL_SECRET, _event_kinds, _git, _pin_route, _provision,
    _reservation, _reset_fake, _run_cleanup_retry, host, sdk,
)

DESIGNER = "raphael-designer"
#: Seats admitted before the designer: (profile, card owned_paths, scope of its machine).
ROLE_CARDS = [
    (sd.WORKER_PROFILE, ["."], ["."]),
    ("raphael-builder", ["app.py"], ["app.py"]),
    ("raphael-verifier", [], []),
    ("raphael-planner", None, []),
]
ROLES = [profile for profile, _card, _scope in ROLE_CARDS]
CARDS = [(profile, card) for profile, card, _scope in ROLE_CARDS]
PATCH = b"--- a/app.py\n+++ b/app.py\n@@ -1 +1 @@\n-print('hi')\n+print('designed')\n"


def _seat(host, monkeypatch, profile, owned_paths):
    """Make the host card ``profile``'s own card and run this session as ``profile``."""
    scope = None if owned_paths is None else json.dumps(owned_paths)
    with kb.connect_closing() as conn, kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET assignee=?, owned_paths=? WHERE id=?",
            (profile, scope, host.task_id),
        )
        conn.execute("UPDATE task_runs SET profile=? WHERE id=?", (profile, host.run_id))
    monkeypatch.setenv("HERMES_PROFILE", profile)


@pytest.fixture
def credential_calls(host, monkeypatch):
    """The host-credential functions the tool path called, in order."""
    calls = []
    for name in (
        "_prepare_host_credential", "_resolve_host_credential",
        "_claude_account_credentials",
    ):
        def spy(*args, _real=getattr(sd, name), _name=name, **kwargs):
            calls.append(_name)
            return _real(*args, **kwargs)

        monkeypatch.setattr(sd, name, spy)
    return calls


def _provision_code(caplog, args=None):
    """Run the tool; return its JSON and the refusal code it logged, if any."""
    caplog.clear()
    with caplog.at_level(logging.INFO, logger=sd.logger.name):
        out = _provision(args)
    code = None
    for record in caplog.records:
        message = record.getMessage()
        if record.name != sd.logger.name:
            continue
        if message.startswith("raphael sandbox dispatch refused: "):
            code = message.split(": ", 1)[1]
        elif message.startswith("raphael sandbox dispatch: {"):
            entry = json.loads(message.split(": ", 1)[1])
            code = entry.get("reason") if entry.get("outcome") == "failed" else code
    return out, code


def _assert_refused(host, caplog, credential_calls, code):
    out, logged = _provision_code(caplog)
    assert "error" in out and logged == code, (out, logged)
    assert FakeSandbox.created == []
    assert _reservation(host)["state"] == "absent"
    assert credential_calls == []


def _designer_machine():
    """Provision for the seated designer: its receipt and the machine it got."""
    out = _provision()
    assert "sandbox_id" in out, out
    assert sd.check_provision_available() is True
    return out, FakeSandbox.created[-1]


def _artifact(args):
    return json.loads(sa.handle_artifact(args))


def _assert_no_coding_authority(out, box, credential_calls):
    policy = out["policy"]
    assert box.reported_metadata["hermes_profile"] == DESIGNER
    assert out["ownership_scope"] == []
    assert policy["host_patch_import_authorized"] is False
    assert policy["credential_vault"] is False
    assert policy["credential_proxy"] is False
    assert policy["sandbox_token_is_placeholder"] is False
    assert policy["automatic_run_cleanup"] is True
    kwargs = box.create_kwargs
    assert box.vault_calls == []
    assert kwargs["credential_proxy"] is None
    assert set(kwargs["env"]) <= {"ANTHROPIC_MODEL", "CLAUDE_CODE_EFFORT_LEVEL"}
    assert not {"ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"} & set(kwargs["env"])
    for place in (kwargs, out):
        assert sd.PLACEHOLDER_TOKEN not in repr(place)
        assert REAL_SECRET not in repr(place)
    assert credential_calls == []


# --- 1. The designer gets a build machine, and no coding authority with it ---


def test_designer_is_admitted_read_only_beside_verifier_and_planner():
    assert DESIGNER in sd.SANDBOX_PROFILES
    assert {DESIGNER, "raphael-verifier", "raphael-planner"} <= sd.READ_ONLY_SOURCE_PROFILES


@pytest.mark.parametrize(
    "owned_paths", [None, [], ["."]], ids=["unscoped", "empty", "whole-tree"],
)
def test_designer_gets_a_machine_with_no_coding_authority(
    host, sdk, monkeypatch, credential_calls, owned_paths,
):
    _seat(host, monkeypatch, DESIGNER, owned_paths)
    out, box = _designer_machine()
    _assert_no_coding_authority(out, box, credential_calls)
    assert box.create_kwargs["env"] == {}
    assert _reservation(host)["state"] == "active"


def test_designers_pinned_route_reaches_its_machine_beside_no_credential(
    host, sdk, monkeypatch, credential_calls,
):
    _seat(host, monkeypatch, DESIGNER, [])
    route = task_assignment_for(DESIGNER, "anthropic", "deep")
    _pin_route(
        host, model=route.model, provider=route.provider,
        effort=route.reasoning_effort, tier="deep",
        lock=kb.mint_policy_lock(
            DESIGNER, route.provider, route.model, route.reasoning_effort, "deep",
        ),
    )
    out, box = _designer_machine()
    _assert_no_coding_authority(out, box, credential_calls)
    env = dict(box.create_kwargs["env"])
    assert env.pop("ANTHROPIC_MODEL") == route.model
    assert env.pop("CLAUDE_CODE_EFFORT_LEVEL", route.reasoning_effort) == route.reasoning_effort
    assert env == {}


@pytest.mark.parametrize("how", ["task-completed hook", "dispatch tick retry"])
def test_designers_machine_is_cleaned_up_when_its_run_ends(host, sdk, monkeypatch, how):
    _seat(host, monkeypatch, DESIGNER, [])
    _out, box = _designer_machine()
    with kb.connect_closing() as conn, kb.write_txn(conn):
        conn.execute(
            "UPDATE task_runs SET status='done', outcome='completed', ended_at=1 "
            "WHERE id=?", (host.run_id,),
        )
        conn.execute(
            "UPDATE tasks SET status='done', current_run_id=NULL WHERE id=?",
            (host.task_id,),
        )
    if how == "task-completed hook":
        sd.cleanup_sandbox_on_task_completed(
            task_id=host.task_id, run_id=host.run_id, board="default",
            assignee=DESIGNER, profile_name=DESIGNER,
        )
    else:
        _run_cleanup_retry()
    assert box.killed is True
    assert _reservation(host)["state"] == "released"
    assert _event_kinds(host).count("sandbox_released") == 1


def test_designers_lease_is_renewed_without_touching_a_vault(
    host, sdk, monkeypatch, credential_calls,
):
    _seat(host, monkeypatch, DESIGNER, [])
    _out, box = _designer_machine()
    box.expires_in_seconds = 120
    box.renewals, patched = [], []
    box.renew = lambda timeout: box.renewals.append(timeout.total_seconds())
    box.credential_vault.patch = lambda **kwargs: patched.append(kwargs)
    result = sd.maintain_sandbox_lease(
        tool_name="mcp_opensandbox_agent_factory_sandbox_command_run",
        args={"sandbox_id": box.id},
    )
    assert result is None, result
    assert box.renewals == [sd.SANDBOX_TIMEOUT_SECONDS]
    assert patched == []
    assert credential_calls == []


def test_designer_cannot_bypass_the_tool_gate_with_the_codex_runtime(host, monkeypatch):
    _seat(host, monkeypatch, DESIGNER, [])
    result = sd.enforce_sandbox_runtime(api_mode="codex_app_server")
    assert result and result["action"] == "block", result
    with kb.connect_closing() as conn:
        task = kb.get_task(conn, host.task_id)
    assert (task.status, task.block_kind) == ("blocked", "capability")


# --- 1b. The gate holds in the environment the dispatcher gives its worker ---


def _spawned_env(host, monkeypatch):
    """What the dispatcher's own spawn hands the worker it starts for the host card."""
    (host.home / "profiles" / DESIGNER).mkdir(parents=True, exist_ok=True)
    with kb.connect_closing() as conn:
        task = kb.get_task(conn, host.task_id)
    spawned = {}

    def popen(_cmd, **kwargs):
        spawned.update(kwargs["env"])
        kwargs["stdout"].close()
        return SimpleNamespace(pid=4242)

    with monkeypatch.context() as patch:
        patch.setattr(subprocess, "Popen", popen)
        assert kb._default_spawn(task, str(host.worktree)) == 4242
    return spawned


@pytest.mark.parametrize("name_from", ["HERMES_PROFILE", "profile home"])
def test_designer_is_admitted_in_the_env_the_dispatcher_injects(
    host, sdk, monkeypatch, name_from,
):
    _seat(host, monkeypatch, DESIGNER, [])
    env = _spawned_env(host, monkeypatch)
    assert {"HERMES_KANBAN_DB", "HERMES_KANBAN_BOARD", "HERMES_KANBAN_WORKSPACE"} <= set(env)
    assert env["HERMES_KANBAN_TASK"] == host.task_id
    assert env["HERMES_KANBAN_RUN_ID"] == str(host.run_id)
    if name_from == "profile home":
        del env["HERMES_PROFILE"]
    for key in set(os.environ) | set(env):
        if key not in env:
            monkeypatch.delenv(key, raising=False)
        elif os.environ.get(key) != env[key]:
            monkeypatch.setenv(key, env[key])
    assert Path(os.environ["HERMES_HOME"]) == host.home / "profiles" / DESIGNER
    assert ("HERMES_PROFILE" in os.environ) is (name_from == "HERMES_PROFILE")
    _out, box = _designer_machine()
    assert box.reported_metadata["hermes_profile"] == DESIGNER


# --- 2. Admission is per seat: only the designer's own card, only its own run ---


@pytest.mark.parametrize(("role", "owned_paths"), CARDS, ids=ROLES)
def test_designer_cannot_provision_for_another_roles_card(
    host, sdk, monkeypatch, caplog, credential_calls, role, owned_paths,
):
    _seat(host, monkeypatch, role, owned_paths)
    monkeypatch.setenv("HERMES_PROFILE", DESIGNER)
    _assert_refused(host, caplog, credential_calls, "task_not_assigned")
    _seat(host, monkeypatch, DESIGNER, [])
    _designer_machine()


def test_a_delegated_designer_cannot_provision(
    host, sdk, monkeypatch, caplog, credential_calls,
):
    _seat(host, monkeypatch, DESIGNER, [])
    with delegated_child_context("designer-child"):
        assert sd.check_provision_available() is False
        _assert_refused(host, caplog, credential_calls, "not_dispatcher_owned")
    _designer_machine()


@pytest.mark.parametrize("role", ROLES)
def test_another_role_cannot_provision_for_the_designers_card(
    host, sdk, monkeypatch, caplog, credential_calls, role,
):
    _seat(host, monkeypatch, DESIGNER, [])
    monkeypatch.setenv("HERMES_PROFILE", role)
    _assert_refused(host, caplog, credential_calls, "task_not_assigned")
    monkeypatch.setenv("HERMES_PROFILE", DESIGNER)
    _designer_machine()


# --- 3. No patch authority: the kernel refuses a designer patch, the tree stays ---


@pytest.mark.parametrize(
    ("owned_paths", "refusal"),
    [([], "read-only runs cannot materialize"), (None, "persisted base and owned_paths")],
    ids=["empty-scope", "unscoped"],
)
def test_kernel_refuses_a_designer_patch_and_leaves_the_worktree_alone(
    host, sdk, monkeypatch, owned_paths, refusal,
):
    _seat(host, monkeypatch, DESIGNER, owned_paths)
    with kb.connect_closing() as conn:
        kb.set_branch_name(conn, host.task_id, _git(host.worktree, "branch", "--show-current"))
        kb.record_worktree_base(conn, host.task_id, host.worktree)
    _out, box = _designer_machine()
    box.files.write_file("/workspace/task/designer.patch", PATCH)
    exported = _artifact({
        "direction": "export", "path": "/workspace/task/designer.patch",
        "expected_sha256": hashlib.sha256(PATCH).hexdigest(),
    })
    assert "attachment_id" in exported, exported
    handoff = {
        "patch_attachment_id": exported["attachment_id"],
        "expected_run_id": host.run_id, "fire_lifecycle_hook": False,
    }
    with kb.connect_closing() as conn:
        with pytest.raises(kb.WorktreeScopeError, match=refusal):
            kb.complete_task(conn, host.task_id, summary="Designed.", **handoff)
        assert kb.get_task(conn, host.task_id).status == "running"
    assert _git(host.worktree, "rev-parse", "HEAD") == host.head
    assert _git(host.worktree, "status", "--porcelain") == ""
    # The same bytes land for a coding seat: the refusal above is the designer's seat.
    _seat(host, monkeypatch, sd.WORKER_PROFILE, ["."])
    with kb.connect_closing() as conn:
        assert kb.complete_task(conn, host.task_id, summary="Applied.", **handoff)
        task = kb.get_task(conn, host.task_id)
        assert task.head_commit != task.base_commit


# --- 3c. The artifact tool is no wider for the designer than for any read-only role ---


def test_designer_moves_its_own_artifacts_through_its_machine(host, sdk, monkeypatch):
    _seat(host, monkeypatch, DESIGNER, [])
    brief = b"Option A is calm. Option B is bold.\n"
    digest = hashlib.sha256(brief).hexdigest()
    with kb.connect_closing() as conn:
        brief_id = kb.store_attachment_bytes(
            conn, host.task_id, "brief.txt", brief,
            content_type="text/plain", uploaded_by="operator",
        )
    _out, box = _designer_machine()

    seen = _artifact({"direction": "inspect", "attachment_id": brief_id})
    assert seen["sha256"] == digest
    assert seen["content_text"] == brief.decode("utf-8")
    imported = _artifact({
        "direction": "import", "attachment_id": brief_id, "expected_sha256": digest,
    })
    assert imported["path"] == f"/workspace/inputs/{brief_id}/brief.txt", imported
    assert box.files.read_bytes(imported["path"]) == brief

    picture = bytes(range(256)) * 9
    box.files.write_file("/workspace/task/option-a.png", picture)
    box.files.write_file("/tmp/elsewhere.png", picture)
    request = {
        "direction": "export", "path": "/workspace/task/option-a.png",
        "expected_sha256": hashlib.sha256(picture).hexdigest(),
    }
    exported = _artifact(request)
    assert exported.get("task_id") == host.task_id, exported
    with kb.connect_closing() as conn:
        stored = kb.get_attachment(conn, exported["attachment_id"])
        assert (stored.task_id, stored.uploaded_by) == (host.task_id, "agent")
        assert kb.read_attachment_bytes(stored) == picture
        before = [item.id for item in kb.list_attachments(conn, host.task_id)]
    refused = _artifact({**request, "path": "/tmp/elsewhere.png"})
    assert refused.get("error") == "export path must be inside this task's remote workspace"
    with kb.connect_closing() as conn:
        assert [item.id for item in kb.list_attachments(conn, host.task_id)] == before


@pytest.mark.parametrize(
    ("project", "refusal"),
    [
        (None, "this task has no project-bound artifact inputs"),
        ("project-a", "attachment is not an approved input of this project task"),
    ],
    ids=["no-project", "same-project-not-a-predecessor"],
)
def test_designer_cannot_reach_another_cards_attachment(
    host, sdk, monkeypatch, project, refusal,
):
    _seat(host, monkeypatch, DESIGNER, [])
    data = b"belongs to somebody else's card"
    with kb.connect_closing() as conn:
        other = kb.create_task(conn, title="Somebody else's card", assignee=sd.WORKER_PROFILE)
        foreign_id = kb.store_attachment_bytes(
            conn, other, "plan.txt", data, content_type="text/plain", uploaded_by="operator",
        )
        if project:
            with kb.write_txn(conn):
                conn.execute(
                    "UPDATE tasks SET project_id=? WHERE id IN (?, ?)",
                    (project, host.task_id, other),
                )
        before = [item.id for item in kb.list_attachments(conn, host.task_id)]
    _out, box = _designer_machine()
    digest = hashlib.sha256(data).hexdigest()
    for request in (
        {"direction": "inspect", "attachment_id": foreign_id},
        {"direction": "copy", "attachment_id": foreign_id, "expected_sha256": digest},
        {"direction": "import", "attachment_id": foreign_id, "expected_sha256": digest},
    ):
        assert _artifact(request).get("error") == refusal, request
    assert not [path for path in box.file_data if path.startswith("/workspace/inputs/")]
    with kb.connect_closing() as conn:
        assert [item.id for item in kb.list_attachments(conn, host.task_id)] == before


# --- 4. Every role admitted before behaves as before ---


@pytest.mark.parametrize(("profile", "owned_paths", "scope"), ROLE_CARDS, ids=ROLES)
def test_each_earlier_role_keeps_exactly_its_authority(
    host, sdk, monkeypatch, credential_calls, profile, owned_paths, scope,
):
    _seat(host, monkeypatch, profile, owned_paths)
    out = _provision()
    assert "sandbox_id" in out, out
    box = FakeSandbox.created[-1]
    has_credential = profile == sd.WORKER_PROFILE
    policy = out["policy"]
    assert box.reported_metadata["hermes_profile"] == profile
    assert out["ownership_scope"] == scope
    assert policy["host_patch_import_authorized"] is bool(scope)
    assert policy["credential_vault"] is has_credential
    assert policy["credential_proxy"] is has_credential
    assert bool(box.vault_calls) is has_credential
    assert (box.create_kwargs["credential_proxy"] is not None) is has_credential
    assert (sd.PLACEHOLDER_TOKEN in repr(box.create_kwargs["env"])) is has_credential
    assert bool(credential_calls) is has_credential
    assert policy["automatic_run_cleanup"] is True


def test_admission_sets_keep_their_relations():
    assert {"raphael-verifier", "raphael-planner"} <= sd.READ_ONLY_SOURCE_PROFILES
    assert not {sd.WORKER_PROFILE, "raphael-builder"} & sd.READ_ONLY_SOURCE_PROFILES
    assert sd.READ_ONLY_SOURCE_PROFILES <= sd.SANDBOX_PROFILES
    assert sd.WORKER_PROFILE in sd.SANDBOX_PROFILES


@pytest.mark.parametrize("profile", ["default", "raphael-business"])
def test_profiles_never_admitted_are_still_refused(host, sdk, monkeypatch, caplog, profile):
    monkeypatch.setenv("HERMES_PROFILE", profile)
    assert sd.check_provision_available() is False
    out, code = _provision_code(caplog)
    assert "error" in out and code == "wrong_profile", (out, code)
    assert FakeSandbox.created == []
