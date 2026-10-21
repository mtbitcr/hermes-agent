"""Card A2: the owner's delivery list beside the reviewed policy.

The reviewed policy file stays read-only source. A second list, kept under the
shared kanban root, holds the repositories the owner confirmed: only a write
carrying the digest the owner was shown fills it, the reviewed entry always
wins, and a list that cannot be read counts as empty. Every test imports the
new module itself, so each one fails on its own while the module is missing.
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pytest

WORKER = "test-worker"
EXAMPLE = "mtbitcr/example-app"
QUALITY = ["quality"]

_WORKER_ENV = (
    "HERMES_KANBAN_DB",
    "HERMES_KANBAN_BOARD",
    "HERMES_KANBAN_TASK",
    "HERMES_KANBAN_RUN_ID",
    "HERMES_KANBAN_HOME",
    "HERMES_KANBAN_WORKSPACES_ROOT",
    "HERMES_SESSION_ID",
)


@pytest.fixture
def root(tmp_path, monkeypatch):
    """A Hermes root standing in for ~/.hermes, outside any worker environment."""
    root = tmp_path / ".hermes"
    root.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(root))
    for name in _WORKER_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return root


def _owner_files(root: Path) -> tuple[Path, Path]:
    """The owner list and its confirmations, located from the root by hand."""
    delivery = root / "kanban" / "delivery"
    return delivery / "owner_policy.json", delivery / "owner_policy_confirmations.jsonl"


def _snapshot(paths) -> list:
    return [path.read_bytes() if path.exists() else None for path in paths]


def _code_policy():
    """The reviewed policy, read from its file the way the kernel reads it."""
    from hermes_cli.kanban_delivery import _POLICY_FILE
    from hermes_cli.kanban_delivery_fences import load_policy

    return load_policy(_POLICY_FILE.read_text(encoding="utf-8"))


def _confirm(owner, repository: str, checks: list):
    """The owner confirms exactly the entry, and the digest of it, that they were shown."""
    return owner.record_confirmed_entry(repository, checks, owner.entry_digest(repository, checks))


def _write_list(root: Path, text: str) -> Path:
    """An owner list written by hand, as a damaged or edited file would be."""
    listed, _ = _owner_files(root)
    listed.parent.mkdir(parents=True, exist_ok=True)
    listed.write_text(text, encoding="utf-8")
    return listed


def test_without_an_owner_list_the_policy_is_the_reviewed_one(root):
    from hermes_cli import kanban_delivery as kd
    from hermes_cli import kanban_delivery_owner_policy as owner

    code = _code_policy()
    assert kd._policy_repositories() == frozenset(code.required_checks)
    assert owner.owner_policy().required_checks == {}
    assert owner.combined_policy(code) == code
    assert not (root / "kanban" / "delivery").exists()


def test_a_confirmed_owner_entry_adds_a_third_repository(root):
    from hermes_cli import kanban_delivery as kd
    from hermes_cli import kanban_delivery_owner_policy as owner
    from hermes_cli.kanban_delivery_fences import load_policy

    code = _code_policy()
    before = int(time.time())
    decision = _confirm(owner, EXAMPLE, QUALITY)
    after = int(time.time())
    assert (decision.allowed, decision.code) == (True, "confirmed")

    assert kd._policy_repositories() == frozenset(code.required_checks) | {EXAMPLE}
    assert len(kd._policy_repositories()) == len(code.required_checks) + 1
    assert owner.owner_policy().required_checks == {EXAMPLE: ("quality",)}
    assert owner.combined_policy(code).required_checks == {
        **code.required_checks, EXAMPLE: ("quality",),
    }

    # The list has exactly the reviewed policy's shape; the confirmation line
    # names the entry, the digest the owner was shown and the time.
    listed, confirmations = _owner_files(root)
    assert load_policy(listed.read_text(encoding="utf-8")).required_checks == {
        EXAMPLE: ("quality",),
    }
    [line] = confirmations.read_text(encoding="utf-8").splitlines()
    record = json.loads(line)
    assert sorted(record) == ["confirmed_at", "digest", "repository", "required_checks"]
    assert (record["repository"], record["required_checks"]) == (EXAMPLE, QUALITY)
    assert record["digest"] == owner.entry_digest(EXAMPLE, QUALITY)
    assert before <= record["confirmed_at"] <= after


def test_a_reviewed_repository_is_refused_and_keeps_its_checks(root):
    from hermes_cli import kanban_delivery as kd
    from hermes_cli import kanban_delivery_owner_policy as owner

    code = _code_policy()
    platform = "mtbitcr/hermes-agent"
    assert platform in code.required_checks
    # GitHub names are case-insensitive, so another spelling is the same repository.
    for spelling in (platform, "mtbitcr/Hermes-Agent"):
        assert owner.validate_entry(spelling, QUALITY, code).code == "platform_repository"
        decision = _confirm(owner, spelling, QUALITY)
        assert (decision.allowed, decision.code) == (False, "platform_repository")
    assert _snapshot(_owner_files(root)) == [None, None]

    # Written into the list by hand, it still never changes the reviewed checks.
    _write_list(root, json.dumps({
        platform: {"required_checks": QUALITY},
        "mtbitcr/Hermes-Agent": {"required_checks": QUALITY},
    }))
    assert owner.owner_policy().required_checks == {}
    assert owner.combined_policy(code) == code
    assert owner.combined_policy(code).required_checks[platform] == code.required_checks[platform]
    assert kd._policy_repositories() == frozenset(code.required_checks)


def test_a_foreign_account_or_wider_checks_are_refused(root):
    from hermes_cli import kanban_delivery_owner_policy as owner

    code = _code_policy()
    foreign = _confirm(owner, "other/app", QUALITY)
    assert (foreign.allowed, foreign.code) == (False, "foreign_account")
    wider = _confirm(owner, EXAMPLE, ["quality", "x"])
    assert (wider.allowed, wider.code) == (False, "checks_not_allowed")
    assert owner.validate_entry("other/app", QUALITY, code).code == "foreign_account"
    assert owner.validate_entry(EXAMPLE, ["quality", "x"], code).code == "checks_not_allowed"
    assert owner.validate_entry("not a repository", QUALITY, code).code == "invalid_repository"
    assert _snapshot(_owner_files(root)) == [None, None]

    # Written into the list by hand, only the valid entry is read.
    _write_list(root, json.dumps({
        "other/app": {"required_checks": QUALITY},
        "mtbitcr/wider-app": {"required_checks": ["quality", "x"]},
        EXAMPLE: {"required_checks": QUALITY},
    }))
    assert owner.owner_policy().required_checks == {EXAMPLE: ("quality",)}


def test_a_wrong_shown_digest_records_nothing(root):
    from hermes_cli import kanban_delivery_owner_policy as owner

    files = _owner_files(root)
    wrong = owner.record_confirmed_entry(EXAMPLE, QUALITY, "0" * 64)
    assert (wrong.allowed, wrong.code) == (False, "digest_mismatch")
    assert _snapshot(files) == [None, None]

    assert _confirm(owner, EXAMPLE, QUALITY).code == "confirmed"
    before = _snapshot(files)
    # The digest shown for one entry does not confirm another.
    stale = owner.record_confirmed_entry(
        "mtbitcr/second-app", QUALITY, owner.entry_digest(EXAMPLE, QUALITY),
    )
    assert (stale.allowed, stale.code) == (False, "digest_mismatch")
    assert _snapshot(files) == before
    assert owner.owner_policy().required_checks == {EXAMPLE: ("quality",)}


def test_a_repeat_confirmation_appends_no_second_line(root):
    from hermes_cli import kanban_delivery_owner_policy as owner

    assert _confirm(owner, EXAMPLE, QUALITY).code == "confirmed"
    files = _owner_files(root)
    before = _snapshot(files)
    repeat = _confirm(owner, EXAMPLE, QUALITY)
    assert (repeat.allowed, repeat.code) == (True, "already_confirmed")
    assert _snapshot(files) == before
    assert len(files[1].read_text(encoding="utf-8").splitlines()) == 1


def test_an_unreadable_owner_list_counts_as_empty(root, monkeypatch):
    from hermes_cli import kanban_delivery as kd
    from hermes_cli import kanban_delivery_owner_policy as owner

    code = _code_policy()
    entry = json.dumps({"required_checks": QUALITY})
    valid = f'{{"{EXAMPLE}": {entry}}}'
    listed = _write_list(root, valid)
    # Control: this list is read.
    assert kd._policy_repositories() == frozenset(code.required_checks) | {EXAMPLE}

    for text in (
        valid[:-1],  # corrupt JSON
        f'{{"{EXAMPLE}": {entry}, "{EXAMPLE}": {entry}}}',  # a duplicate key
        f'{{"{EXAMPLE}": {{"required_checks": ["quality"], "note": "x"}}}}',  # an extra field
        "[" * 100_000,  # nested too deep to parse
    ):
        listed.write_text(text, encoding="utf-8")
        assert owner.owner_policy().required_checks == {}, text[:80]
        assert owner.combined_policy(code) == code
        assert kd._policy_repositories() == frozenset(code.required_checks)

    listed.write_bytes(b'{"\xff": 1}')  # not UTF-8
    assert owner.owner_policy().required_checks == {}
    assert kd._policy_repositories() == frozenset(code.required_checks)

    # Should the owner list's module itself raise, the kernel reads the reviewed policy alone.
    listed.write_text(valid, encoding="utf-8")
    assert kd._policy_repositories() == frozenset(code.required_checks) | {EXAMPLE}

    def _unresolvable():
        raise RuntimeError("the kanban root cannot be resolved")

    monkeypatch.setattr(owner, "owner_policy_path", _unresolvable)
    assert kd._policy_repositories() == frozenset(code.required_checks)


def _spawned_worker_env(kb, task, workspace: Path, board: str, monkeypatch) -> dict:
    """The environment kanban_db._default_spawn gives the worker it launches.

    Only the final process launch is replaced: the stand-in records the argv
    and environment the dispatcher built and returns a pid, so no process
    starts. SQLite stays real.
    """
    launches = []

    class _Launched:
        pid = 4242

    def _launch(argv, **kwargs):
        log = kwargs.get("stdout")
        if hasattr(log, "close"):
            log.close()
        launches.append((list(argv), dict(kwargs.get("env") or {})))
        return _Launched()

    with monkeypatch.context() as patch:
        patch.setattr(subprocess, "Popen", _launch)
        pid = kb._default_spawn(task, str(workspace), board=board)
    assert pid == _Launched.pid
    assert len(launches) == 1, launches
    argv, env = launches[0]
    assert f"work kanban task {task.id}" in argv
    return env


def _become(env: dict, monkeypatch) -> None:
    """Carry on as the worker: this process now has exactly ``env``."""
    for name in list(os.environ):
        if name not in env:
            monkeypatch.delenv(name)
    for name, value in env.items():
        if os.environ.get(name) != value:
            monkeypatch.setenv(name, value)


_READ_AS_DEFAULT_PROFILE = """
import json
from hermes_cli import kanban_delivery as kd
from hermes_cli import kanban_delivery_owner_policy as owner
print(json.dumps({
    "path": str(owner.owner_policy_path()),
    "owner": {name: list(checks) for name, checks in owner.owner_policy().required_checks.items()},
    "repositories": sorted(kd._policy_repositories()),
}))
"""


def test_store_path_under_the_dispatcher_worker_environment(root, tmp_path, monkeypatch):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_delivery as kd
    from hermes_cli import kanban_delivery_owner_policy as owner

    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    kb.create_board("proj-a")
    kb.create_board("other")
    own_db = root / "kanban" / "boards" / "proj-a" / "kanban.db"
    other_db = root / "kanban" / "boards" / "other" / "kanban.db"
    profile_home = root / "profiles" / WORKER
    profile_home.mkdir(parents=True)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    code_repositories = frozenset(_code_policy().required_checks)

    # The dispatcher claims a card on its board and spawns through the live _default_spawn.
    conn = kb.connect(board="proj-a")
    try:
        tid = kb.create_task(conn, title="connect a repository", assignee=WORKER)
        claimed = kb.claim_task(conn, tid)
        assert claimed is not None
        env = _spawned_worker_env(kb, claimed, workspace, "proj-a", monkeypatch)
    finally:
        conn.close()
    assert env["HERMES_KANBAN_DB"] == str(own_db)
    assert env["HERMES_KANBAN_BOARD"] == "proj-a"
    assert env["HERMES_KANBAN_TASK"] == tid
    assert env["HERMES_HOME"] == str(profile_home)

    # The worker carries on with exactly that environment: a write and a read
    # resolve the one list under the shared root.
    _become(env, monkeypatch)
    listed, confirmations = _owner_files(root)
    assert owner.owner_policy_path() == listed
    assert owner.confirmations_path() == confirmations
    assert _confirm(owner, EXAMPLE, QUALITY).code == "confirmed"
    assert owner.owner_policy().required_checks == {EXAMPLE: ("quality",)}
    assert kd._policy_repositories() == code_repositories | {EXAMPLE}

    # From there the root registry and every other board still resolve.
    assert kb.kanban_home() == root
    assert kb.register_db_path() == root / "kanban" / "board_register.db"
    register = sqlite3.connect(root / "kanban" / "board_register.db")
    try:
        registered = {name for (name,) in register.execute("SELECT board_name FROM board_register")}
    finally:
        register.close()
    on_disk = {
        child.name for child in (root / "kanban" / "boards").iterdir()
        if (child / "kanban.db").exists()
    }
    assert registered == on_disk == {"proj-a", "other"}
    assert {entry["slug"] for entry in kb.list_boards()} == on_disk | {"default"}
    assert all(kb.get_register_entry(name) is not None for name in registered)
    assert kb.board_dir("other") / "kanban.db" == other_db

    # The tool's own answer, checked against an independent count of the files.
    entries = json.loads(listed.read_text(encoding="utf-8"))
    lines = confirmations.read_text(encoding="utf-8").splitlines()
    assert (list(entries), len(lines)) == ([EXAMPLE], 1)
    assert len(kd._policy_repositories()) == len(code_repositories) + len(entries)
    assert sorted(path.name for path in listed.parent.iterdir()) == [listed.name, confirmations.name]
    assert not (profile_home / "kanban").exists()

    # A second process living in the default profile home reads the same entries.
    child_env = {name: value for name, value in os.environ.items() if not name.startswith("HERMES_")}
    child_env.update(HERMES_HOME=str(root), HOME=str(tmp_path))
    result = subprocess.run(
        [sys.executable, "-c", _READ_AS_DEFAULT_PROFILE],
        cwd=Path(__file__).resolve().parents[2],
        env=child_env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip().splitlines()[-1]) == {
        "path": str(listed),
        "owner": {EXAMPLE: QUALITY},
        "repositories": sorted(code_repositories | {EXAMPLE}),
    }
