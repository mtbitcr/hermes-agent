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

The seat is admitted only on a card that owns no repository path. The kernel's
read-only run policy does not name the designer yet, so a designer card that
owns paths gets no machine at all (``designer_card_has_owned_paths``): no
machine means no export, and no export means no patch to commit.
"""
from __future__ import annotations

import contextlib
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
#: The one code, and the one sentence, for a designer card that owns repository paths.
DESIGNER_OWNED_PATHS = "designer_card_has_owned_paths"
DESIGNER_OWNED_PATHS_REPLY = (
    "a designer card that owns repository paths gets no build machine, "
    "because a designer has no patch authority."
)
#: Card scopes that own a repository path: one file, and the whole tree.
OWNING_SCOPES = [["app.py"], ["."]]
OWNING_IDS = ["file", "whole-tree"]


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


def _set_owned_paths(host, owned_paths):
    """Change only the host card's owned paths; the session and its environment stay as they are."""
    scope = None if owned_paths is None else json.dumps(owned_paths)
    with kb.connect_closing() as conn, kb.write_txn(conn):
        conn.execute("UPDATE tasks SET owned_paths=? WHERE id=?", (scope, host.task_id))


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


@pytest.fixture
def board_writes(monkeypatch):
    """The two calls a provision writes for a machine through: its creation intent and its reservation."""
    calls = []
    for owner, name in ((kb, "record_kernel_creation"), (sd, "_advance_reservation")):
        def spy(*args, _real=getattr(owner, name), _name=name, **kwargs):
            calls.append(_name)
            return _real(*args, **kwargs)

        monkeypatch.setattr(owner, name, spy)
    return calls


def _refusal_code(caplog):
    """The code the tool logged for a refused call, if it logged one."""
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
    return code


def _provision_code(caplog, args=None):
    """Run the tool; return its JSON and the refusal code it logged, if any."""
    caplog.clear()
    with caplog.at_level(logging.INFO, logger=sd.logger.name):
        out = _provision(args)
    return out, _refusal_code(caplog)


def _assert_refused(host, caplog, credential_calls, code):
    out, logged = _provision_code(caplog)
    assert "error" in out and logged == code, (out, logged)
    assert FakeSandbox.created == []
    assert _reservation(host)["state"] == "absent"
    assert credential_calls == []
    return out


def _board_state(host):
    """What the board and the creation ledger hold for the host card, to compare around a refusal."""
    with kb.connect_closing() as conn:
        events = [
            tuple(row) for row in conn.execute(
                "SELECT id, run_id, kind, payload FROM task_events "
                "WHERE task_id = ? ORDER BY id", (host.task_id,),
            )
        ]
        attachments = [item.id for item in kb.list_attachments(conn, host.task_id)]
    return {
        "events": events,
        "attachments": attachments,
        "intents": kb.kernel_resource_intents(kind=sd.KERNEL_SANDBOX_KIND, task_id=host.task_id),
        "reservation": _reservation(host),
    }


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


@pytest.mark.parametrize("owned_paths", [None, []], ids=["unscoped", "empty"])
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


def _worker_env(host, monkeypatch, name_from):
    """Run this session in the exact environment the dispatcher hands the worker it starts."""
    board_db, board = kb.kanban_db_path(), kb.get_current_board()
    env = _spawned_env(host, monkeypatch)
    assert {"HERMES_KANBAN_DB", "HERMES_KANBAN_BOARD", "HERMES_KANBAN_WORKSPACE"} <= set(env)
    assert Path(env["HERMES_KANBAN_DB"]) == board_db
    assert env["HERMES_KANBAN_BOARD"] == board
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


@pytest.mark.parametrize("name_from", ["HERMES_PROFILE", "profile home"])
def test_designer_is_admitted_in_the_env_the_dispatcher_injects(
    host, sdk, monkeypatch, name_from,
):
    _seat(host, monkeypatch, DESIGNER, [])
    _worker_env(host, monkeypatch, name_from)
    _out, box = _designer_machine()
    assert box.reported_metadata["hermes_profile"] == DESIGNER


@pytest.mark.parametrize("name_from", ["HERMES_PROFILE", "profile home"])
def test_designer_card_with_owned_paths_is_refused_in_the_env_the_dispatcher_injects(
    host, sdk, monkeypatch, caplog, credential_calls, name_from,
):
    _seat(host, monkeypatch, DESIGNER, ["app.py"])
    _worker_env(host, monkeypatch, name_from)
    out = _assert_refused(host, caplog, credential_calls, DESIGNER_OWNED_PATHS)
    assert out["error"] == DESIGNER_OWNED_PATHS_REPLY
    # Same environment, same card: with its owned paths gone it is granted its machine.
    _set_owned_paths(host, [])
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


@pytest.mark.parametrize(
    ("how", "code"),
    [
        ("delegated child", "not_dispatcher_owned"),
        ("another role's session", "task_not_assigned"),
        ("stale run", "run_not_active"),
    ],
)
def test_older_refusals_keep_their_codes_on_a_designer_card_that_owns_paths(
    host, sdk, monkeypatch, caplog, credential_calls, how, code,
):
    _seat(host, monkeypatch, DESIGNER, ["app.py"])
    if how == "another role's session":
        monkeypatch.setenv("HERMES_PROFILE", sd.WORKER_PROFILE)
    elif how == "stale run":
        monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(host.run_id + 90))
    child = (
        delegated_child_context("designer-child")
        if how == "delegated child" else contextlib.nullcontext()
    )
    with child:
        _assert_refused(host, caplog, credential_calls, code)


# --- 2b. A designer card that owns repository paths gets no machine at all ---


@pytest.mark.parametrize("owned_paths", OWNING_SCOPES, ids=OWNING_IDS)
def test_designer_card_with_owned_paths_gets_no_machine_and_nothing_is_written(
    host, sdk, monkeypatch, caplog, credential_calls, board_writes, owned_paths,
):
    _seat(host, monkeypatch, DESIGNER, owned_paths)
    before = _board_state(host)
    out = _assert_refused(host, caplog, credential_calls, DESIGNER_OWNED_PATHS)
    assert out["error"] == DESIGNER_OWNED_PATHS_REPLY
    assert "sandbox_id" not in out
    assert FakeSandbox.connect_calls == []
    assert board_writes == []
    assert _board_state(host) == before
    # Not vacuous: the same card owning nothing is granted its machine, and that does write.
    _set_owned_paths(host, [])
    _designer_machine()
    assert {"record_kernel_creation", "_advance_reservation"} <= set(board_writes)
    after = _board_state(host)
    assert after["events"] != before["events"]
    assert before["intents"] == [] and len(after["intents"]) == 1


@pytest.mark.parametrize("owned_paths", OWNING_SCOPES, ids=OWNING_IDS)
def test_designer_card_with_owned_paths_has_nothing_to_export(
    host, sdk, monkeypatch, caplog, credential_calls, owned_paths,
):
    _seat(host, monkeypatch, DESIGNER, owned_paths)
    _assert_refused(host, caplog, credential_calls, DESIGNER_OWNED_PATHS)
    before = _board_state(host)["attachments"]
    request = {
        "direction": "export", "path": "/workspace/task/designer.patch",
        "expected_sha256": hashlib.sha256(PATCH).hexdigest(),
    }
    exported = _artifact(request)
    assert "error" in exported and "attachment_id" not in exported, exported
    assert FakeSandbox.created == [] and FakeSandbox.connect_calls == []
    assert _board_state(host)["attachments"] == before
    # Not vacuous: the same request is served once the card owns nothing and so gets its machine.
    _set_owned_paths(host, [])
    _out, box = _designer_machine()
    box.files.write_file(request["path"], PATCH)
    assert "attachment_id" in _artifact(request)


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


def _tool(name, args):
    """Call one model tool the way the agent loop does: through the registry, with its middleware."""
    import model_tools

    return json.loads(model_tools.handle_function_call(name, args))


def _claimed_designer_card(host, monkeypatch, owned_paths):
    """A designer card as the dispatcher makes it: created with a pinned route, claimed, on the host worktree."""
    route = task_assignment_for(DESIGNER, "anthropic", "deep")
    for profile, provider in ((DESIGNER, "anthropic"), ("raphael-verifier", "openai-codex")):
        configured = task_assignment_for(profile, provider, "deep")
        home = host.home / "profiles" / profile
        home.mkdir(parents=True, exist_ok=True)
        (home / "config.yaml").write_text(json.dumps({
            "model": {"provider": configured.provider, "default": configured.model},
            "agent": {"reasoning_effort": configured.reasoning_effort},
            "fallback_providers": [], "fallback_model": None,
        }), encoding="utf-8")
    branch = _git(host.worktree, "branch", "--show-current")
    with kb.connect_closing() as conn:
        assert kb.block_task(conn, host.task_id, reason="replaced by a claimed designer card")
        task_id = kb.create_task(
            conn, title="Designer owns this option rendering card", assignee=DESIGNER,
            workspace_kind="worktree", workspace_path=str(host.worktree),
            branch_name=branch, owned_paths=owned_paths,
            provider_override=route.provider, model_override=route.model,
            reasoning_effort=route.reasoning_effort, execution_tier="deep",
            model_policy_lock=kb.mint_policy_lock(
                DESIGNER, route.provider, route.model, route.reasoning_effort, "deep",
            ),
            risk_tier=2,
        )
        kb.recompute_ready(conn)
        claimed = kb.claim_task(conn, task_id, expected_assignee=DESIGNER)
        assert claimed is not None
        kb.record_worktree_base(conn, task_id, host.worktree)
        assert kb.get_task(conn, task_id).owned_paths == owned_paths
    monkeypatch.setenv("HERMES_PROFILE", DESIGNER)
    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(claimed.current_run_id))
    return task_id, branch


def _source_state(host, branch):
    """The source the host holds: every commit, plus the card's branch and worktree while they exist.

    Finishing a clean card removes its worktree and ``wt/`` branch, so their absence is not a change.
    """
    state = {"commits": set(_git(host.repo, "rev-list", "--all").split())}
    if _git(host.repo, "branch", "--list", branch):
        state["branch_head"] = _git(host.repo, "rev-parse", branch)
        state["branch_source"] = _git(host.repo, "show", f"{branch}:app.py")
    if host.worktree.exists():
        state["worktree_head"] = _git(host.worktree, "rev-parse", "HEAD")
        state["worktree_source"] = (host.worktree / "app.py").read_text(encoding="utf-8")
        state["worktree_status"] = _git(host.worktree, "status", "--porcelain")
    return state


@pytest.mark.parametrize("handoff", ["explicit-complete", "saved-complete", "request-review"])
@pytest.mark.parametrize(
    "owned_paths", [[], None, *OWNING_SCOPES], ids=["empty", "unscoped", *OWNING_IDS],
)
def test_no_designer_patch_is_committed_through_any_handoff_tool(
    host, sdk, monkeypatch, caplog, owned_paths, handoff,
):
    task_id, branch = _claimed_designer_card(host, monkeypatch, owned_paths)
    before = _source_state(host, branch)

    caplog.clear()
    with caplog.at_level(logging.INFO, logger=sd.logger.name):
        receipt = _tool("raphael_sandbox_provision", {})
    refusal = _refusal_code(caplog)
    patch_id = 999_999  # a refused card has no machine to export from, so nothing real to cite
    if "sandbox_id" in receipt:
        FakeSandbox.created[-1].files.write_file("/workspace/task/designer.patch", PATCH)
        exported = _tool("raphael_sandbox_artifact", {
            "direction": "export", "path": "/workspace/task/designer.patch",
            "expected_sha256": hashlib.sha256(PATCH).hexdigest(),
        })
        assert "attachment_id" in exported, exported
        patch_id = exported["attachment_id"]

    args = {"task_id": task_id, "summary": "Rendered the options."}
    if handoff == "explicit-complete":
        args["patch_attachment_id"] = patch_id
    tool = "kanban_request_review" if handoff == "request-review" else "kanban_complete"
    reply = _tool(tool, args)

    after = _source_state(host, branch)
    assert after == {key: before[key] for key in after}, (receipt, reply)
    if handoff == "explicit-complete":
        assert "error" in reply, reply
    if owned_paths:
        assert "error" in receipt and "sandbox_id" not in receipt, receipt
        assert refusal == DESIGNER_OWNED_PATHS
        assert FakeSandbox.created == []
        with kb.connect_closing() as conn:
            uploads = kb.list_attachments(conn, task_id)
        assert [item.id for item in uploads if item.uploaded_by == "agent"] == []
    else:
        assert "sandbox_id" in receipt, receipt
        assert receipt["ownership_scope"] == []
        assert receipt["policy"]["host_patch_import_authorized"] is False


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


@pytest.mark.parametrize("owned_paths", OWNING_SCOPES, ids=OWNING_IDS)
@pytest.mark.parametrize("profile", ROLES)
def test_a_card_that_owns_paths_is_refused_for_the_designer_alone(
    host, sdk, monkeypatch, caplog, profile, owned_paths,
):
    _seat(host, monkeypatch, profile, owned_paths)
    out, code = _provision_code(caplog)
    assert "sandbox_id" in out and code != DESIGNER_OWNED_PATHS, (out, code)
    writes = profile in (sd.WORKER_PROFILE, "raphael-builder")
    assert out["ownership_scope"] == (owned_paths if writes else [])


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
