"""Tests for the journeys seed (card 2): seed_states.py and seed_stack.sh.

seed_states.py runs for real against a temporary platform home: its commands are the real hermes
commands, its suggestion and give-up come from the real kernel, and the owner-workspace reads
(decisions list, board read) are the real ones. The reads need the Project to be committed by the
owner, which the seed never does; the tests play that commit with the platform's own receipt functions.
seed_stack.sh runs against a stand-in docker that sits first on PATH: it records every call, answers
inspect, and runs the snippets and the seed program that are sent into the dashboard on this machine.
No docker and no network is involved.

The shape of seed.json is not described anywhere in this repository. The seed assumes the narrowest
one the plan supports, and SEED below is that shape: a top-level "prefix", the "project" name, and per
journey number the names of its cards by the state each one is made in.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from agent import estop
from hermes_cli import kanban_db, owner_workspace as ow, projects_db, release_ledger
from hermes_cli.config import load_config

pytestmark = pytest.mark.linux_only

REPO = Path(__file__).resolve().parents[2]
JOURNEYS = REPO / "scripts" / "journeys"
SEED_STATES = JOURNEYS / "seed_states.py"
SEED_STACK = JOURNEYS / "seed_stack.sh"
TEMPLATE = JOURNEYS / "stack.env.template"
STACK_HOME = JOURNEYS / "stack-home"

BOARD = "journey-test"
PROJECT_SLUG = "journey-test-project"
PROJECT_NAME = "Journey test project"
QUESTION = "Journey test question"
SUGGESTION = "Journey test suggestion"
STOPPED = "Journey test stopped card"

SEED = {
    "prefix": "Journey test",
    "project": PROJECT_NAME,
    "journeys": {
        "2": {"review": "Journey test card waiting for review"},
        "3": {"suggestion": SUGGESTION, "question": QUESTION},
        "5": {"stopped": STOPPED},
    },
}
SEED_TEXT = json.dumps(SEED, indent=2) + "\n"

# Where the released owner app copy keeps seed.json inside the browser container.
SEED_FILE = "/opt/raphael-journeys/tools/page-check/journeys/seed.json"

# Bytecode of the platform modules that the seed's commands import is cached here, never in the tree.
REAL_PYC = Path(tempfile.gettempdir()) / f"journeys-seed-pyc-{os.getuid()}"  # windows-footgun: ok - the module is linux_only

# The docker --format shapes that seed_stack.sh asks for (the same ones reset_stack.sh uses).
F_STATE = '{{.State.Running}} {{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}'
F_DEST = '{{range .Mounts}}{{println .Name .Destination}}{{end}}'
F_HOME = '{{range .Config.Env}}{{$p := split . "="}}{{if eq (index $p 0) "HERMES_HOME"}}{{index $p 1}}{{end}}{{end}}'

MODEL_TOKEN = "sk-ant-oat01-SENTINEL-model-token-0123456789"


def template_names(section: str) -> list[str]:
    names, current = [], None
    for raw in TEMPLATE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        header = re.fullmatch(r"\[([^\]]+)\]", line)
        if header:
            current = header.group(1)
        elif current == section:
            names.append(line)
    return names


def tree(root: Path) -> dict[str, str]:
    """Relative path -> content hash, for proving a folder was not touched."""
    seen: dict[str, str] = {}
    for base, dirs, files in os.walk(root):
        for name in sorted(dirs) + sorted(files):
            full = Path(base) / name
            seen[str(full.relative_to(root))] = "dir" if full.is_dir() else hashlib.sha256(full.read_bytes()).hexdigest()
    return seen


def count(db: Path, sql: str) -> int:
    with contextlib.closing(sqlite3.connect(f"file:{db}?mode=ro", uri=True)) as conn:
        return conn.execute(sql).fetchone()[0]


@pytest.fixture
def seed(monkeypatch):
    """seed_states.py imported for real; the commands it runs find the platform and keep bytecode in /tmp."""
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    monkeypatch.setenv("PYTHONPATH", str(REPO))
    monkeypatch.setenv("PYTHONPYCACHEPREFIX", str(REAL_PYC))
    spec = importlib.util.spec_from_file_location("journeys_seed_states", SEED_STATES)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def reset_home() -> Path:
    """A freshly reset stack home: what reset_stack.sh writes from stack-home, and nothing else."""
    home = Path(os.environ["HERMES_HOME"])
    shutil.copytree(STACK_HOME, home, dirs_exist_ok=True)
    return home


def run_seed(seed, capsys, journey: str, text: str = SEED_TEXT) -> tuple[int, str, str]:
    code = seed.main([journey], text)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def commit_project(slug: str) -> None:
    """Play the owner's commit of the Project: a committed receipt and the board's owner stamp."""
    ctx = ow.resolve_owner_context()
    key = f"commit-{slug}"
    with contextlib.closing(projects_db.connect()) as conn:
        ow._ensure_schema(conn)
        project = projects_db.get_project(conn, slug)
        _, _, token = ow._acquire_or_replay(conn, ctx, key, "owner_task_graph_commit", "d" * 64)
        ow._update_progress(conn, ctx, key, token, project_id=project.id, board_slug=project.board_slug)
        ow._finalize_receipt(
            conn, ctx, key, token, status="committed",
            result={"ok": True, "project_id": project.id, "root_task_id": "none", "task_ids": []},
        )
    kanban_db.write_board_metadata(project.board_slug, project_id=project.id)


# --- 1. Journey 3: one waiting question and one waiting suggestion, no release record.


def test_journey_3_leaves_one_question_and_one_suggestion_waiting_for_the_owner(seed, reset_home, capsys):
    code, out, err = run_seed(seed, capsys, "3")
    assert code == 0, err

    commit_project(PROJECT_SLUG)
    decisions = ow.list_owner_decisions(ow.resolve_owner_context())["data"]

    assert sorted((d["kind"], d["authority"], d["title"], d["project_name"]) for d in decisions) == [
        ("capability", "recommendation", SUGGESTION, PROJECT_NAME),
        ("owner_input", "task", QUESTION, PROJECT_NAME),
    ]
    assert not release_ledger.ledger_path().exists()


# --- 2. Journey 5: a card stopped after repeated failures, the dispatcher off, no worker.


def test_journey_5_leaves_a_stopped_card_with_the_dispatcher_off_and_starts_no_worker(
    seed, reset_home, capsys, monkeypatch,
):
    def a_worker_was_started(*args, **kwargs):
        raise AssertionError("the seed started a worker")

    monkeypatch.setattr(kanban_db, "_default_spawn", a_worker_was_started)
    code, out, err = run_seed(seed, capsys, "5")
    assert code == 0, err
    assert "seed_states: dispatcher setting: off\n" in out

    commit_project(PROJECT_SLUG)
    ctx = ow.resolve_owner_context()
    snapshot = ow.read_project_snapshot(ctx, PROJECT_SLUG)
    shown = [(column["name"], task) for column in snapshot["columns"] for task in column["tasks"]]
    assert [task["title"] for _, task in shown] == [STOPPED]
    column, stopped = shown[0]
    assert column == "blocked"
    assert stopped["stopped_work"] == "gave_up"
    assert isinstance(stopped["event_revision"], int) and stopped["event_revision"] > 0

    with contextlib.closing(kanban_db.connect(board=BOARD)) as conn:
        card = kanban_db.get_task(conn, kanban_db.list_tasks(conn)[0].id)
        assert card.worker_pid is None and card.current_run_id is None

    # The dispatcher setting reads off in the copied config, and the seed left it as the reset wrote it.
    assert load_config()["kanban"]["dispatch_in_gateway"] is False
    assert (reset_home / "config.yaml").read_text(encoding="utf-8") == (STACK_HOME / "config.yaml").read_text(encoding="utf-8")

    # What "work dispatcher off" can mean: the owner-workspace answer carries one execution state, and it
    # moves with neither candidate. Not with the global pause, and not with the gateway setting.
    execution = snapshot["steward"]["execution"]
    estop.engage()
    assert estop.is_engaged()
    assert ow.read_project_snapshot(ctx, PROJECT_SLUG)["steward"]["execution"] == execution
    estop.disengage()
    config = reset_home / "config.yaml"
    config.write_text(config.read_text(encoding="utf-8").replace("dispatch_in_gateway: false", "dispatch_in_gateway: true"), encoding="utf-8")
    assert load_config()["kanban"]["dispatch_in_gateway"] is True
    assert ow.read_project_snapshot(ctx, PROJECT_SLUG)["steward"]["execution"] == execution


# --- 3. Refusals: each prints one fixed line, exits 1 and writes nothing.


def _without_journey_5(text: str) -> str:
    data = json.loads(text)
    del data["journeys"]["5"]
    return json.dumps(data)


def _with_question(name: str) -> str:
    data = json.loads(SEED_TEXT)
    data["journeys"]["3"]["question"] = name
    return json.dumps(data)


def _a_card_is_already_on_the_board() -> None:
    kanban_db.create_board(BOARD)
    with contextlib.closing(kanban_db.connect(board=BOARD)) as conn:
        kanban_db.create_task(conn, title="A card that was already there")


def _the_dispatcher_setting_is_on() -> None:
    config = Path(os.environ["HERMES_HOME"]) / "config.yaml"
    config.write_text(
        config.read_text(encoding="utf-8").replace("dispatch_in_gateway: false", "dispatch_in_gateway: true"), encoding="utf-8"
    )


REFUSALS = {
    "name-without-the-prefix": ("3", _with_question("Quarterly SENTINEL-NAME question"), None),
    "journey-seed-json-lacks": ("5", _without_journey_5(SEED_TEXT), None),
    "journey-2": ("2", SEED_TEXT, None),
    "card-already-on-the-board": ("3", SEED_TEXT, _a_card_is_already_on_the_board),
    "dispatcher-setting-not-off": ("5", SEED_TEXT, _the_dispatcher_setting_is_on),
}


@pytest.mark.parametrize("case", sorted(REFUSALS))
def test_the_seed_refuses_with_one_fixed_line_and_writes_nothing(seed, reset_home, capsys, case):
    journey, text, set_up = REFUSALS[case]
    if set_up:
        set_up()
    before = tree(reset_home)

    code, out, err = run_seed(seed, capsys, journey, text)

    assert code == 1
    assert out == ""
    assert len(err.strip().splitlines()) == 1
    assert "SENTINEL-NAME" not in err
    # Nothing was written: not one file of the home changed, and no project, no new board, no new card.
    assert tree(reset_home) == before
    assert (reset_home / "kanban" / "boards" / BOARD).exists() is (case == "card-already-on-the-board")
    assert not (reset_home / "projects.db").exists()
    if case == "card-already-on-the-board":
        with contextlib.closing(kanban_db.connect(board=BOARD)) as conn:
            assert [task.title for task in kanban_db.list_tasks(conn)] == ["A card that was already there"]


# --- 4. seed_stack.sh: the dashboard container, the home owner's ids, nothing secret.

HOME_OWNER = "10001:10002"

FAKE_DOCKER = r'''#!PYTHON_PATH
import json, os, subprocess, sys

state = json.load(open(os.environ["FAKE_DOCKER_STATE"]))
argv = sys.argv[1:]


def record(stdin=None):
    with open(state["log"], "a") as fh:
        fh.write(json.dumps({"argv": argv, "stdin": stdin}) + "\n")


def refuse(why):
    sys.stderr.write("fake docker: " + why + "\n")
    sys.exit(98)


def show(done):
    sys.stdout.write(done.stdout)
    sys.stderr.write(done.stderr)
    sys.exit(done.returncode)


if argv[:1] == ["inspect"]:
    record()
    if argv[1:4] != ["--type", "container", "--format"] or len(argv) != 6:
        refuse("inspect needs --type container --format FORMAT NAME")
    fmt, name = argv[4], argv[5]
    entry = state["containers"].get(name)
    if entry is None:
        sys.stderr.write("Error: No such container: " + name + "\n")
        sys.exit(1)
    if fmt == state["fmt_state"]:
        print("true none" if entry["running"] else "false none")
    elif fmt == state["fmt_dest"]:
        for mount in entry["mounts"]:
            print(mount["name"], mount["dest"])
    elif fmt == state["fmt_home"]:
        print(entry["home"])
    else:
        refuse("format not supported by the stand-in")
    sys.exit(0)

if argv[:1] != ["exec"]:
    record()
    refuse("only inspect and exec are allowed")

args, interactive = argv[1:], False
while args and args[0].startswith("-"):
    if args[0] == "-i":
        interactive, args = True, args[1:]
    elif args[0] == "--user":
        args = args[2:]
    else:
        record()
        refuse("exec option not supported by the stand-in")
container, cmd = args[0], args[1:]
data = sys.stdin.read() if interactive else None
record(data)
entry = state["containers"].get(container)
if entry is None:
    refuse("no such container")
if container == state["app"] and cmd in (["id", "-u"], ["id", "-g"]):
    print(state["app_user"].split(":")[0 if cmd[1] == "-u" else 1])
elif container == state["dashboard"] and cmd[:3] == ["stat", "-c", "%u:%g"]:
    print(state["home_owner"])
elif container == state["browser"] and len(cmd) == 2 and cmd[0] == "cat" and cmd[1] in state["files"]:
    sys.stdout.write(state["files"][cmd[1]])
elif container == state["dashboard"] and cmd[:2] == ["sh", "-c"]:
    show(subprocess.run(["sh", "-c", *cmd[2:]], env={"PATH": state["container_path"]}, capture_output=True, text=True, encoding="utf-8"))
elif container == state["dashboard"] and cmd[1:2] == ["-c"]:
    env = {
        "PATH": "/usr/bin:/bin", "HOME": state["scratch"], "HERMES_HOME": entry["host_home"],
        "PYTHONPATH": state["repo"], "PYTHONPYCACHEPREFIX": state["pyc"],
    }
    show(subprocess.run(cmd, input=data, env=env, capture_output=True, text=True, encoding="utf-8", cwd=state["repo"]))
else:
    refuse("command not supported by the stand-in")
'''


def split_exec(argv: list[str]) -> dict:
    args, user, interactive = argv[1:], None, False
    while args[0].startswith("-"):
        if args[0] == "-i":
            interactive, args = True, args[1:]
        else:
            assert args[0] == "--user", args
            user, args = args[1], args[2:]
    return {"user": user, "interactive": interactive, "container": args[0], "cmd": args[1:]}


def stand_in_stack(base: Path, app_user: str) -> tuple[dict, Path, Path]:
    """Five stand-in containers: the dashboard's home is a folder on this machine; docker and hostname are stand-ins."""
    proxy, app, dash, gate, browser = template_names("containers")
    base.mkdir(parents=True)
    platform_home = base / "dashboard-home"
    shutil.copytree(STACK_HOME, platform_home)
    (platform_home / ".env").write_text(f"API_SERVER_KEY={MODEL_TOKEN}\n", encoding="utf-8")
    bin_dir = base / "bin"
    bin_dir.mkdir()
    # The dashboard's PATH: the image's privilege-drop shim first, then the venv launcher, whose first line
    # names the platform's Python.
    shim = base / "container" / "opt" / "hermes" / "bin"
    venv = base / "container" / "opt" / "hermes" / ".venv" / "bin"
    shim.mkdir(parents=True)
    venv.mkdir(parents=True)
    (shim / "hermes").write_text('#!/bin/sh\nexec /opt/hermes/.venv/bin/hermes "$@"\n', encoding="utf-8")
    (venv / "hermes").write_text(f"#!{sys.executable}\nraise SystemExit(0)\n", encoding="utf-8")
    log = base / "calls.jsonl"
    state = {
        "log": str(log), "fmt_state": F_STATE, "fmt_dest": F_DEST, "fmt_home": F_HOME,
        "app": app, "dashboard": dash, "browser": browser, "home_owner": HOME_OWNER, "app_user": app_user,
        "container_path": f"{shim}:{venv}:/usr/bin:/bin", "scratch": str(base), "repo": str(REPO),
        "pyc": str(REAL_PYC), "files": {SEED_FILE: SEED_TEXT},
        "containers": {
            name: {"running": True, "mounts": [], "home": "", "host_home": ""} for name in (proxy, app, dash, gate, browser)
        },
    }
    state["containers"][dash].update(
        mounts=[{"name": template_names("volumes")[0], "dest": "/opt/data"}], home="/opt/data", host_home=str(platform_home)
    )
    state_file = base / "state.json"
    state_file.write_text(json.dumps(state), encoding="utf-8")
    (bin_dir / "docker").write_text(FAKE_DOCKER.replace("PYTHON_PATH", sys.executable), encoding="utf-8")
    (bin_dir / "hostname").write_text("#!/bin/sh\necho journeys-host.invalid\n", encoding="utf-8")
    for name in ("docker", "hostname"):
        (bin_dir / name).chmod(0o755)
    env = {
        "PATH": f"{bin_dir}:{os.environ['PATH']}", "HOME": str(base), "FAKE_DOCKER_STATE": str(state_file),
        "JOURNEY_STACK_HOST": "journeys-host.invalid", "CLAUDE_CODE_OAUTH_TOKEN": MODEL_TOKEN, "API_SERVER_KEY": MODEL_TOKEN,
    }
    return env, log, platform_home


def run_stack_script(env: dict, base: Path, journey: str) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(SEED_STACK), journey], env=env, cwd=base, capture_output=True, text=True, encoding="utf-8", timeout=240)


def test_seed_stack_runs_in_the_dashboard_as_the_home_owner_and_keeps_secrets_off_command_lines(tmp_path):
    proxy, app, dash, gate, browser = template_names("containers")
    env, log, platform_home = stand_in_stack(tmp_path / "stack", app_user="1000:1000")

    done = run_stack_script(env, tmp_path, "5")

    assert done.returncode == 0, done.stderr
    calls = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert {call["argv"][0] for call in calls} <= {"inspect", "exec"}
    execs = [split_exec(call["argv"]) | {"stdin": call["stdin"]} for call in calls if call["argv"][0] == "exec"]

    # The seed program: in the dashboard (the third entry of [containers]), as the home owner, with seed.json's
    # text on standard input and the journey number as its one argument.
    program = [e for e in execs if e["cmd"][:1] == [sys.executable]]
    assert len(program) == 1
    assert program[0]["container"] == dash
    assert program[0]["interactive"] is True
    assert program[0]["cmd"][0] == sys.executable and program[0]["cmd"][-1] == "5"
    assert program[0]["stdin"].strip() == SEED_TEXT.strip()

    # The only commands without the home owner's ids are the lookups of the ids themselves. Every other command
    # runs as the account that owns the dashboard's home folder: never root, never the owner app's user.
    lookups = {(e["container"], tuple(e["cmd"])) for e in execs if e["user"] is None}
    assert lookups == {(app, ("id", "-u")), (dash, ("stat", "-c", "%u:%g", "/opt/data"))}
    assert all(e["user"] == HOME_OWNER for e in execs if e["user"] is not None)
    assert {e["container"] for e in execs} == {app, dash, browser}
    assert [e["cmd"] for e in execs if e["container"] == browser] == [["cat", SEED_FILE]]
    assert not any(name in json.dumps(calls) for name in (proxy, gate))

    # Nothing secret is on a command line, on standard input or in the output.
    assert MODEL_TOKEN not in json.dumps(calls) + done.stdout + done.stderr

    # The program really ran in the home it was sent to: the stopped card is there.
    assert count(platform_home / "kanban" / "boards" / BOARD / "kanban.db", "SELECT COUNT(*) FROM tasks") == 1

    # A home folder that belongs to the owner app's own user is refused, and no program is started.
    env2, log2, home2 = stand_in_stack(tmp_path / "same-user", app_user=HOME_OWNER)
    refused = run_stack_script(env2, tmp_path, "5")
    assert refused.returncode == 1
    assert len(refused.stderr.strip().splitlines()) == 1
    calls2 = [json.loads(line) for line in log2.read_text(encoding="utf-8").splitlines()]
    assert not [c for c in calls2 if c["argv"][0] == "exec" and split_exec(c["argv"])["cmd"][:1] == [sys.executable]]
    assert not (home2 / "kanban" / "boards" / BOARD).exists()


# --- 5. The store paths: run from inside a worker, the seed still reaches the root's stores.


def test_the_seed_reaches_the_journey_board_and_the_root_stores_from_inside_a_worker(tmp_path, reset_home):
    root = reset_home
    kanban_db.create_board("worker-board")
    with contextlib.closing(kanban_db.connect(board="worker-board")) as conn:
        worker_task = kanban_db.create_task(conn, title="The worker's own card")
    with contextlib.closing(projects_db.connect()) as conn:
        projects_db.create_project(conn, name="An older project", slug="an-older-project")
    with contextlib.closing(release_ledger.connect()):
        pass
    profile_home = root / "profiles" / "raphael-executor"
    profile_home.mkdir(parents=True)

    env = {
        "PATH": os.environ["PATH"], "HOME": str(tmp_path), "PYTHONPATH": str(REPO), "PYTHONPYCACHEPREFIX": str(REAL_PYC),
        "HERMES_HOME": str(profile_home), "HERMES_PROFILE": "raphael-executor",
        "HERMES_KANBAN_DB": str(root / "kanban" / "boards" / "worker-board" / "kanban.db"),
        "HERMES_KANBAN_BOARD": "worker-board", "HERMES_KANBAN_TASK": worker_task,
        "HERMES_KANBAN_WORKSPACES_ROOT": str(root / "kanban" / "boards" / "worker-board" / "workspaces"),
        "HERMES_KANBAN_WORKSPACE": str(tmp_path), "HERMES_KANBAN_RUN_ID": "1", "HERMES_KANBAN_CLAIM_LOCK": "host:1",
    }
    done = subprocess.run(
        [sys.executable, str(SEED_STATES), "3"], input=SEED_TEXT, env=env, cwd=tmp_path,
        capture_output=True, text=True, encoding="utf-8", timeout=240,
    )
    assert done.returncode == 0, done.stderr

    reported = dict(re.findall(r"^seed_states: (.+): (\S+)$", done.stdout, re.M))
    journey_db = root / "kanban" / "boards" / BOARD / "kanban.db"
    registry = root / "projects.db"
    worker_db = root / "kanban" / "boards" / "worker-board" / "kanban.db"

    # What the seed reported against what the stores hold, counted here and not by the seed.
    assert int(reported["work cards on the journey board"]) == count(journey_db, "SELECT COUNT(*) FROM tasks WHERE task_kind = 'work'") == 1
    assert int(reported["projects in the root registry"]) == count(registry, "SELECT COUNT(*) FROM projects") == 2
    suggestions = count(journey_db, f"SELECT COUNT(*) FROM tasks WHERE task_kind = 'recommendation' AND id = '{reported['suggestion']}'")
    assert suggestions == 1
    assert count(journey_db, "SELECT COUNT(*) FROM tasks") == 2
    assert reported["release record"] == "present"
    assert (root / "kanban" / "release_ledger.db").exists()

    # Nothing went to the worker's own board or to its profile.
    assert count(worker_db, "SELECT COUNT(*) FROM tasks") == 1
    assert not (profile_home / "projects.db").exists()
    assert not (profile_home / "kanban").exists()
