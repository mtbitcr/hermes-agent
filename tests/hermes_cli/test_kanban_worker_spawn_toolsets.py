from __future__ import annotations

import subprocess

import pytest


def _make_task(kb, *, assignee: str):
    return kb.Task(
        id="t_spawn_tools",
        title="spawn tools",
        body=None,
        assignee=assignee,
        status="running",
        priority=0,
        created_by="test",
        created_at=1,
        started_at=None,
        completed_at=None,
        workspace_kind="dir",
        workspace_path=None,
        claim_lock="lock",
        claim_expires=None,
        tenant=None,
        current_run_id=7,
    )


def test_default_spawn_pins_assignee_profile_cli_toolsets(monkeypatch, tmp_path):
    """Manual profile assignment should keep that profile's CLI tools.

    Regression guard for dispatcher-spawned workers that boot with
    HERMES_KANBAN_TASK: the worker must not collapse to only kanban lifecycle
    tools when the assigned profile's top-level ``toolsets`` is the default
    composite. The spawned CLI gets an explicit --toolsets pin resolved from
    platform_toolsets.cli; model_tools appends task-scoped kanban tools later.
    """
    root = tmp_path / ".hermes"
    profile = root / "profiles" / "elias"
    profile.mkdir(parents=True)
    profile.joinpath("config.yaml").write_text(
        """
platform_toolsets:
  cli:
    - clarify
    - code_execution
    - delegation
    - file
    - memory
    - session_search
    - skills
    - terminal
    - web
toolsets:
  - hermes-cli
agent:
  disabled_toolsets: []
""".lstrip(),
        encoding="utf-8",
    )
    root.joinpath("config.yaml").write_text("toolsets:\n  - kanban\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))

    from hermes_cli import kanban_db as kb

    monkeypatch.setattr(kb, "_resolve_hermes_argv", lambda: ["hermes"])

    captured = {}

    class FakeProc:
        pid = 4242

    def fake_popen(cmd, *args, **kwargs):
        captured["cmd"] = list(cmd)
        captured["env"] = dict(kwargs.get("env") or {})
        captured["cwd"] = kwargs.get("cwd")
        return FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    pid = kb._default_spawn(_make_task(kb, assignee="elias"), str(workspace))

    assert pid == 4242
    assert captured["env"]["HERMES_HOME"] == str(profile)
    assert captured["env"]["HERMES_KANBAN_TASK"] == "t_spawn_tools"
    assert "--toolsets" in captured["cmd"]
    pinned = captured["cmd"][captured["cmd"].index("--toolsets") + 1].split(",")
    for required in ("terminal", "web", "file", "skills", "code_execution", "delegation"):
        assert required in pinned


def test_default_spawn_model_override_survives_real_cli_parse(monkeypatch, tmp_path):
    """The dispatcher's pre-``chat`` model flag must reach ``args.model``.

    This is an integration contract between Kanban's worker argv builder and
    the real CLI parser. A parser default once erased the explicit override,
    silently sending the worker to its profile default or fallback instead.
    """
    root = tmp_path / ".hermes"
    (root / "profiles" / "elias").mkdir(parents=True)
    root.joinpath("config.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))

    from hermes_cli import kanban_db as kb
    from hermes_cli._parser import build_top_level_parser

    monkeypatch.setattr(kb, "_resolve_hermes_argv", lambda: ["hermes"])
    captured = {}

    class FakeProc:
        pid = 4244

    def fake_popen(cmd, *args, **kwargs):
        captured["cmd"] = list(cmd)
        return FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    task = _make_task(kb, assignee="elias")
    task.model_override = "gpt-5.6-sol"
    kb._default_spawn(task, str(workspace))

    parser, _subparsers, _chat_parser = build_top_level_parser()
    # Profile selection is attached by the outer CLI bootstrap rather than
    # build_top_level_parser(); remove that already-validated prefix and parse
    # the worker flags/subcommand through the real shared parser.
    assert captured["cmd"][1:3] == ["-p", "elias"]
    args = parser.parse_args(captured["cmd"][3:])

    assert args.command == "chat"
    assert args.model == "gpt-5.6-sol"
    assert args.query == "work kanban task t_spawn_tools"


_PINNED_ASSIGNEE = "raphael-builder"


def _spawn_profile_home(monkeypatch, tmp_path):
    root = tmp_path / ".hermes"
    (root / "profiles" / "elias").mkdir(parents=True)
    (root / "profiles" / _PINNED_ASSIGNEE).mkdir(parents=True)
    root.joinpath("config.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return workspace


def _pinned_task(kb, *, assignee: str = _PINNED_ASSIGNEE):
    """A task locked to the deep Claude lane its assignee is approved for."""
    task = _make_task(kb, assignee=assignee)
    task.model_override = "claude-opus-5"
    task.provider_override = "anthropic"
    task.reasoning_effort = "max"
    task.execution_tier = "deep"
    task.model_policy_lock = kb.mint_policy_lock(
        _PINNED_ASSIGNEE, "anthropic", "claude-opus-5", "max", "deep",
    )
    return task


def test_default_spawn_disables_fallbacks_for_a_policy_locked_task(monkeypatch, tmp_path):
    """A pinned task's worker must never be able to switch route.

    Two channels: the env var (inherited by anything the worker itself spawns)
    and ``--no-fallbacks`` on the worker's own argv. The argv flag is the
    AUTHORITY, because the profile's ``.env`` is loaded with override=True
    during startup and could reset the env var — see
    ``test_spawned_worker_keeps_fallbacks_disabled_against_dotenv_override``.
    """
    from hermes_cli import kanban_db as kb
    from hermes_cli.fallback_config import (
        FALLBACKS_DISABLED_ENV,
        NO_FALLBACK_FLAG,
        get_fallback_chain,
    )

    workspace = _spawn_profile_home(monkeypatch, tmp_path)
    monkeypatch.setattr(kb, "_resolve_hermes_argv", lambda: ["hermes"])
    captured = {}

    class FakeProc:
        pid = 4246

    def fake_popen(cmd, *args, **kwargs):
        captured["cmd"] = list(cmd)
        captured["env"] = dict(kwargs.get("env") or {})
        return FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    assert kb._default_spawn(_pinned_task(kb), str(workspace)) == 4246

    assert captured["env"][FALLBACKS_DISABLED_ENV] == "1"
    cmd = captured["cmd"]
    assert cmd[1:3] == ["-p", _PINNED_ASSIGNEE]
    assert NO_FALLBACK_FLAG in cmd
    assert cmd[cmd.index("-m") + 1] == "claude-opus-5"
    assert cmd[cmd.index("--provider") + 1] == "anthropic"
    assert cmd[cmd.index("--reasoning") + 1] == "max"

    # Either channel empties the chain, even against a config that names one.
    drifted = {"fallback_providers": [{"provider": "openai", "model": "gpt-x"}]}
    assert get_fallback_chain(drifted)
    monkeypatch.setenv(FALLBACKS_DISABLED_ENV, "1")
    assert get_fallback_chain(drifted) == []


def test_default_spawn_deep_tier_forwards_max_turns_after_chat(monkeypatch, tmp_path):
    """A deep task gets the dispatcher's iteration budget as a ``chat`` flag.

    ``--max-turns`` belongs to the chat subparser, so it must follow ``chat``;
    the real parser proves the value reaches ``args.max_turns``.
    """
    from hermes_cli import kanban_db as kb
    from hermes_cli._parser import build_top_level_parser

    workspace = _spawn_profile_home(monkeypatch, tmp_path)
    monkeypatch.setattr(kb, "_resolve_hermes_argv", lambda: ["hermes"])
    captured = {}

    class FakeProc:
        pid = 4247

    def fake_popen(cmd, *args, **kwargs):
        captured["cmd"] = list(cmd)
        return FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    kb._default_spawn(_pinned_task(kb), str(workspace))

    cmd = captured["cmd"]
    assert cmd.index("--max-turns") > cmd.index("chat")
    assert cmd[cmd.index("--max-turns") + 1] == str(kb._DEEP_TIER_MAX_TURNS)
    parser, _subparsers, _chat_parser = build_top_level_parser()
    args = parser.parse_args(cmd[3:])
    assert args.max_turns == kb._DEEP_TIER_MAX_TURNS


def test_default_spawn_routine_task_keeps_profile_max_turns(monkeypatch, tmp_path):
    from hermes_cli import kanban_db as kb

    workspace = _spawn_profile_home(monkeypatch, tmp_path)
    monkeypatch.setattr(kb, "_resolve_hermes_argv", lambda: ["hermes"])
    captured = {}

    class FakeProc:
        pid = 4248

    def fake_popen(cmd, *args, **kwargs):
        captured["cmd"] = list(cmd)
        return FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    task = _make_task(kb, assignee="elias")
    task.execution_tier = "routine"
    kb._default_spawn(task, str(workspace))

    assert "--max-turns" not in captured["cmd"]


def test_set_worker_pid_records_max_turns_in_spawned_event(monkeypatch, tmp_path):
    from hermes_cli import kanban_db as kb

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="budgeted")
        kb._set_worker_pid(conn, tid, 4249, max_turns=160)
        events = [e for e in kb.list_events(conn, tid) if e.kind == "spawned"]
        assert events and events[-1].payload == {"pid": 4249, "max_turns": 160}
    finally:
        conn.close()


def test_default_spawn_does_not_disable_fallbacks_for_ordinary_tasks(monkeypatch, tmp_path):
    from hermes_cli import kanban_db as kb
    from hermes_cli.fallback_config import FALLBACKS_DISABLED_ENV

    from hermes_cli.fallback_config import NO_FALLBACK_FLAG

    workspace = _spawn_profile_home(monkeypatch, tmp_path)
    monkeypatch.delenv(FALLBACKS_DISABLED_ENV, raising=False)
    monkeypatch.setattr(kb, "_resolve_hermes_argv", lambda: ["hermes"])
    captured = {}

    class FakeProc:
        pid = 4247

    def fake_popen(cmd, *args, **kwargs):
        captured["cmd"] = list(cmd)
        captured["env"] = dict(kwargs.get("env") or {})
        return FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    kb._default_spawn(_make_task(kb, assignee="elias"), str(workspace))

    assert FALLBACKS_DISABLED_ENV not in captured["env"]
    assert NO_FALLBACK_FLAG not in captured["cmd"]


def test_spawned_worker_keeps_fallbacks_disabled_against_dotenv_override(
    monkeypatch, tmp_path
):
    """A REAL spawned process: the profile's .env says 0, the worker still can't switch.

    This is the whole point of the argv channel. ``load_hermes_dotenv`` loads
    the profile's ``.env`` with ``override=True``, so an env-var-only kill
    switch would be reset to ``0`` during startup. The child below replays the
    exact production order — profile ``.env`` loaded first, then
    ``apply_process_fallback_policy()`` on the argv ``_default_spawn`` built —
    and reports what the fallback surfaces actually answer afterwards.
    """
    import json
    import pathlib
    import sys

    from hermes_cli import kanban_db as kb
    from hermes_cli.fallback_config import FALLBACKS_DISABLED_ENV

    workspace = _spawn_profile_home(monkeypatch, tmp_path)
    profile_home = tmp_path / ".hermes" / "profiles" / _PINNED_ASSIGNEE
    # The adversarial .env: it explicitly turns the kill switch back off, and
    # the profile config restores a fallback chain.
    profile_home.joinpath(".env").write_text(
        f"{FALLBACKS_DISABLED_ENV}=0\n", encoding="utf-8"
    )
    profile_home.joinpath("config.yaml").write_text(
        "fallback_providers:\n  - provider: openai\n    model: gpt-x\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(kb, "_resolve_hermes_argv", lambda: ["hermes"])
    captured = {}

    class FakeProc:
        pid = 4248

    def fake_popen(cmd, *args, **kwargs):
        captured["cmd"] = list(cmd)
        captured["env"] = dict(kwargs.get("env") or {})
        return FakeProc()

    # Scoped: the fake Popen exists only to capture the production argv/env.
    # It must be gone before the real child below runs, or ``subprocess.run``
    # would call it instead of spawning anything.
    with monkeypatch.context() as spawn_patch:
        spawn_patch.setattr(subprocess, "Popen", fake_popen)
        kb._default_spawn(_pinned_task(kb), str(workspace))

    child = r"""
import json, os, sys
from hermes_cli.env_loader import load_hermes_dotenv
from pathlib import Path

# Exactly the production order in hermes_cli.main: profile .env FIRST (with
# override=True), the no-fallback latch AFTER.
load_hermes_dotenv(hermes_home=Path(os.environ["HERMES_HOME"]), load_external_secrets=False)
env_after_dotenv = os.environ.get("HERMES_DISABLE_FALLBACKS")

from hermes_cli.fallback_config import (
    apply_process_fallback_policy, fallbacks_disabled, get_fallback_chain,
    strip_no_fallback_flag,
)

latched = apply_process_fallback_policy()
sys.argv = sys.argv[:1] + strip_no_fallback_flag(sys.argv[1:])

from hermes_cli.config import load_config_readonly
from agent import auxiliary_client as aux

print("RESULT" + json.dumps({
    "env_after_dotenv": env_after_dotenv,
    "latched": latched,
    "disabled": fallbacks_disabled(),
    "config_chain": get_fallback_chain(load_config_readonly()),
    "literal_chain": get_fallback_chain(
        {"fallback_providers": [{"provider": "openai", "model": "gpt-x"}]}
    ),
    "argv_clean": "--no-fallbacks" not in sys.argv,
    "aux_paths": [
        aux._try_payment_fallback("anthropic", "summarize")[0] is None,
        aux._try_main_agent_model_fallback("anthropic", "summarize")[0] is None,
        aux._try_configured_fallback_chain("summarize", "anthropic")[0] is None,
        aux._try_main_fallback_chain("summarize", "anthropic")[0] is None,
        aux._resolve_auto_route(None, "summarize")[0] is None,
    ],
}))
"""
    env = dict(captured["env"])
    # Strip the inheritable env channel entirely, so ONLY the argv authority
    # can be what keeps this worker pinned.
    env.pop(FALLBACKS_DISABLED_ENV, None)
    env["PYTHONPATH"] = str(pathlib.Path(kb.__file__).resolve().parent.parent)
    argv = [sys.executable, "-c", child, *captured["cmd"][1:]]
    proc = subprocess.run(
        argv, env=env, capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stderr
    line = next(
        row for row in proc.stdout.splitlines() if row.startswith("RESULT")
    )
    result = json.loads(line[len("RESULT"):])

    # The .env really did win over the environment — and it changed nothing.
    assert result["env_after_dotenv"] == "0"
    assert result["latched"] is True
    assert result["disabled"] is True
    assert result["config_chain"] == []
    assert result["literal_chain"] == []
    assert result["argv_clean"] is True
    assert all(result["aux_paths"]), result["aux_paths"]


def test_default_spawn_refuses_a_locked_task_it_cannot_honor(monkeypatch, tmp_path):
    """An unhonorable pin fails visibly instead of running on another route."""
    from hermes_cli import kanban_db as kb

    workspace = _spawn_profile_home(monkeypatch, tmp_path)
    monkeypatch.setattr(kb, "_resolve_hermes_argv", lambda: ["hermes"])

    def fail_popen(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("a locked task with an unhonorable pin must not spawn")

    monkeypatch.setattr(subprocess, "Popen", fail_popen)

    for field, value, message in (
        ("provider_override", None, "incomplete"),
        ("reasoning_effort", None, "incomplete"),
        ("execution_tier", None, "no admitted authority"),
        ("model_override", "claude-fable-5", "forbidden model"),
        ("reasoning_effort", "ultra", "forbidden reasoning effort"),
        # A single hand-edited column breaks the digest that binds the tuple.
        ("model_override", "claude-sonnet-5", "not the admitted route"),
        ("execution_tier", "routine", "not the admitted route"),
        # Unknown, stale and legacy-truthy authorities are never "unlocked".
        ("model_policy_lock", "bogus:v1:" + "a" * 64, "unknown authority"),
        ("model_policy_lock", "raphael:v99:" + "a" * 64, "stale"),
        ("model_policy_lock", "raphael", "provenance is unreadable"),
    ):
        task = _pinned_task(kb)
        setattr(task, field, value)
        with pytest.raises(RuntimeError, match=message):
            kb._default_spawn(task, str(workspace))

    # The lock is bound to the ROLE too. ``default`` is independently approved
    # for this very provider/model/effort at this very tier, so the route itself
    # still validates — only the digest catches that this lock was not minted
    # for that role.
    reassigned = _pinned_task(kb, assignee="default")
    with pytest.raises(RuntimeError, match="digest does not bind this route"):
        kb._default_spawn(reassigned, str(workspace))


def test_resolve_worker_cli_toolsets_uses_profile_home_not_parent_config(monkeypatch, tmp_path):
    root = tmp_path / ".hermes"
    profile = root / "profiles" / "elias"
    profile.mkdir(parents=True)
    root.joinpath("config.yaml").write_text("platform_toolsets:\n  cli:\n    - kanban\n", encoding="utf-8")
    profile.joinpath("config.yaml").write_text(
        """
platform_toolsets:
  cli:
    - terminal
    - web
toolsets:
  - hermes-cli
""".lstrip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(root))

    from hermes_cli import kanban_db as kb

    resolved = kb._resolve_worker_cli_toolsets(str(profile))

    assert resolved is not None
    assert "terminal" in resolved
    assert "web" in resolved
    assert "kanban" in resolved  # recovered worker lifecycle surface
    assert resolved != ["kanban"]
