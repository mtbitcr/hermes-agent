"""Shared stand-ins, fixtures and helpers of the journeys stack tests (card P-2).

This module holds no test; the test_journeys_stack*.py files import from it.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import NamedTuple

import pytest
import yaml


REPO = Path(__file__).resolve().parents[2]
JOURNEYS = REPO / "scripts" / "journeys"
RESET = JOURNEYS / "reset_stack.sh"
CHECK = JOURNEYS / "isolation_check.sh"
TEMPLATE = JOURNEYS / "stack.env.template"
STACK_HOME = JOURNEYS / "stack-home"
PROXY_SOURCE = JOURNEYS / "synthetic_proxy.py"

PROXY_C = "raphael-journeys-proxy"
APP_C = "raphael-journeys-app"
DASH_C = "raphael-journeys-dashboard"
GATE_C = "raphael-journeys-gateway"
BROWSER_C = "raphael-journeys-browser"
STACK_CONTAINERS = [PROXY_C, APP_C, DASH_C, GATE_C, BROWSER_C]
HOME_V = "raphael-journeys-home"
TOKENS_V = "raphael-journeys-tokens"
PROXY_V = "raphael-journeys-proxy"
STACK_VOLUMES = [HOME_V, TOKENS_V, PROXY_V]
PROD_C = "raphael-workspace"
# The owner app's Project and board. The slug is spelled like the production container's name but is platform
# data, not a docker resource; the reset writes it twice as the platform's account (project, board metadata).
ANCHOR = "raphael-workspace"
ANCHOR_WRITES = 2
# The two writes that need the platform's code, as reset_stack.sh sends them (a copy of its texts, so a change turns the focused tests red).
PROJECT_CREATE = ["hermes", "project", "create", "Raphael Workspace", "--slug", ANCHOR, "--board", ANCHOR]
BOARD_SHELL = """\
p=$PATH:
found=
while [ -n "$p" ]; do
  d=${p%%:*}
  p=${p#*:}
  f=${d:-.}/hermes
  [ -f "$f" ] && [ -x "$f" ] || continue
  first=
  IFS= read -r first < "$f" || true
  case "$first" in "#!"*) ;; *) continue ;; esac
  set -f
  set -- ${first#"#!"}
  set +f
  case "${1##*/}" in sh | bash | dash | ash | ksh | zsh) continue ;; esac
  found=1
  break
done
[ -n "$found" ] || exit 127
exec "$@" -"""
BOARD_PROGRAM = """\
import sys
from hermes_cli import kanban_db, projects_db

slug = "raphael-workspace"
with projects_db.connect_closing() as conn:
    project = projects_db.get_project(conn, slug)
if project is None or project.slug != slug:
    sys.exit("the Project is not in the registry")
kanban_db.write_board_metadata(slug, project_id=project.id)
meta = kanban_db.read_board_metadata(slug)
if not kanban_db.board_exists(slug) or meta.get("slug") != slug or meta.get("project_id") != project.id:
    sys.exit("the board metadata does not name the Project id")
"""
BOARD_WRITE = ["sh", "-c", BOARD_SHELL, "sh"]
# Old rig names (facts file): present on the fake host so that any call naming them is caught.
DECOY_CONTAINERS = [
    "raphael-s2-workspace", "raphael-s2-proxy", "raphael-s2-dashboard",
    "raphael-s2-harness", "raphael-s2-browser",
]
DECOY_VOLUMES = ["raphael-s2-state", "raphael-s2-state-prod", "raphael-s2-pw"]
# Profile folder names are platform data, not container or volume names.
NOT_RESOURCE_NAMES = {"raphael-planner", "raphael-designer"}

# Bytecode of the platform modules that the fake runs for real is cached here (compiling them takes seconds).
REAL_PYC = Path(tempfile.gettempdir()) / f"journeys-stack-pyc-{os.getuid()}"

STACK_HOST = "journeys-host.invalid"
PROD_ORIGIN = "https://prod.example.invalid"
SIGN_IN_REDIRECT = "https://sign-in.example.invalid/start"
SAME_HOST_REDIRECT = "https://prod.example.invalid/home"

# Sentinel secrets: none of these may reach an argv, stdout or stderr.
MODEL_TOKEN = "sk-ant-oat01-SENTINEL-model-token-0123456789"
PROD_ISSUER = "https://prod-issuer-sentinel.example.invalid/issuer"

SURFACE_ENV = {
    "workspace": "HERMES_API_TOKEN_FILE",
    "recommendations": "HERMES_RECOMMENDATIONS_TOKEN_FILE",
    "connections": "HERMES_CONNECTIONS_API_TOKEN_FILE",
    "automations": "HERMES_AUTOMATIONS_API_TOKEN_FILE",
    "models": "HERMES_MODELS_API_TOKEN_FILE",
}
KEY_ENV = {
    "default": "HERMES_EXECUTOR_API_TOKEN_FILE",
    "planner": "HERMES_PLANNER_API_TOKEN_FILE",
}
APP_ENV_NAMES = [
    "HERMES_API_BASE_URL", "HERMES_CONVERSATION_API_BASE_URL", "HERMES_API_TOKEN_FILE",
    "HERMES_RECOMMENDATIONS_TOKEN_FILE", "HERMES_CONNECTIONS_API_TOKEN_FILE",
    "HERMES_AUTOMATIONS_API_TOKEN_FILE", "HERMES_MODELS_API_TOKEN_FILE",
    "HERMES_EXECUTOR_API_TOKEN_FILE", "HERMES_PLANNER_API_TOKEN_FILE", "NODE_EXTRA_CA_CERTS",
    "NODE_ENV", "PORT", "RAPHAEL_ACCESS_AUDIENCE", "RAPHAEL_ACCESS_ISSUER",
    "RAPHAEL_OWNER_MEMBERSHIP_JSON", "RAPHAEL_OWNER_TIME_ZONE", "RAPHAEL_PROJECT_BINDINGS_JSON",
    "RAPHAEL_PUBLIC_ORIGIN", "RAPHAEL_PLANNER_MODEL", "RAPHAEL_EXECUTOR_MODEL",
]
CHECKS = [
    "names-on-template", "dispatcher-off", "cron-lists-empty", "no-messaging-platform",
    "telegram-send-fails", "production-get-refused",
]


def parse_template(path: Path = TEMPLATE) -> dict[str, list[str]]:
    """Sections of the template: plain text, one name per line, # comments, [sections]."""
    sections: dict[str, list[str]] = {}
    current = None
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        header = re.fullmatch(r"\[([^\]]+)\]", line)
        if header:
            current = header.group(1)
            sections.setdefault(current, [])
            continue
        assert current is not None, f"name outside a section: {line!r}"
        sections[current].append(line)
    return sections


def resource_names_in(text: str) -> set[str]:
    """Container or volume style names (raphael-...) found anywhere in a piece of text."""
    found = set(re.findall(r"raphael-[A-Za-z0-9][A-Za-z0-9-]*", text))
    return found - NOT_RESOURCE_NAMES


def sha_tree(root: Path) -> dict[str, str]:
    """Relative path -> mode and content hash, for proving a folder was not touched."""
    seen: dict[str, str] = {}
    for base, dirs, files in os.walk(root):
        for name in sorted(dirs) + sorted(files):
            full = Path(base) / name
            rel = str(full.relative_to(root))
            mode = stat.S_IMODE(full.lstat().st_mode)
            digest = "dir" if full.is_dir() else hashlib.sha256(full.read_bytes()).hexdigest()
            seen[rel] = f"{mode:o}:{digest}"
    return seen


# The shell fakes (sleep, hostname, docker inspect, docker exec printenv) start in about a thousandth of
# the time of a Python fake, so they read this bash-readable view of the state: ready-made docker outputs.
VIEW_SOURCE = r'''

def _q(text):
    return "'" + str(text).replace("'", "'\\''") + "'"


def render_view(state):
    names = lambda items: "".join(item.split("=", 1)[0] + "\n" for item in items)
    lines = ["declare -A IMG ENVN MTYPES MDEST ENVL RUN HEALTH POLLS IMGENV", "HOST=" + _q(state["hostname"])]
    for name, box in state["containers"].items():
        key = _q(name)
        health = box.get("health") or ""
        lines += [
            "IMG[%s]=%s" % (key, _q(box["image"])),
            "ENVN[%s]=%s" % (key, _q(names(box["env"]))),
            "ENVL[%s]=%s" % (key, _q("\n".join(box["env"]))),
            "MTYPES[%s]=%s" % (key, _q("".join("%s %s\n" % (m["type"], m.get("name", "")) for m in box["mounts"]))),
            "MDEST[%s]=%s" % (key, _q("".join("%s %s\n" % (m.get("name", ""), m["dest"]) for m in box["mounts"]))),
            "RUN[%s]=%s" % (key, 1 if box["running"] else 0),
            "HEALTH[%s]=%s" % (key, _q(health)),
            "POLLS[%s]=%d" % (key, box.get("starting_polls", 0) if health else 0),
        ]
    for image, entry in state["images"].items():
        lines.append("IMGENV[%s]=%s" % (_q(image), _q(names(entry["env"]))))
    return "\n".join(lines) + "\n"


def write_view(state, root):
    tmp = root / "state.sh.tmp"
    tmp.write_text(render_view(state))
    os.replace(tmp, root / "state.sh")
    polls = root / "polls"
    if polls.is_dir():
        for leftover in polls.iterdir():
            leftover.unlink()
'''
_view_namespace: dict = {"os": os, "Path": Path}
exec(VIEW_SOURCE, _view_namespace)
write_view = _view_namespace["write_view"]

FAKE_COMMON = r'''
import json, os, re, stat, sys, time
from pathlib import Path

ROOT = Path(os.environ["FAKE_HOST_ROOT"])
STATE = ROOT / "state.json"
LOG = ROOT / "calls.jsonl"
OWNERS = ROOT / "owners.json"


# The fake keeps who owns each entry of a volume in owners.json ("volume/relative/path" -> [uid, gid];
# the folder itself is "volume"), next to the real mode bits of the temporary files. An entry that is
# not listed belongs to root. A non-root user reaches a path only as the owner, group and other bits allow.
def load_owners():
    return json.loads(OWNERS.read_text()) if OWNERS.exists() else {}


def save_owners(table):
    tmp = OWNERS.with_name("owners.json.tmp")
    tmp.write_text(json.dumps(table))
    os.replace(tmp, OWNERS)


def owner_key(path):
    try:
        return str(Path(os.path.abspath(path)).relative_to(ROOT / "volumes"))
    except ValueError:
        return None


def owner_in(table, path):
    return list(table.get(owner_key(path), [0, 0]))


def tree_of(path):
    found = [str(path)]
    for base, dirs, files in os.walk(path):
        found += [os.path.join(base, name) for name in dirs + files]
    return found


def bits_for(uid, gid, path, table):
    mode = stat.S_IMODE(os.lstat(path).st_mode)
    owner_uid, owner_gid = owner_in(table, path)
    if uid == owner_uid:
        return (mode >> 6) & 7
    if gid == owner_gid:
        return (mode >> 3) & 7
    return mode & 7


def reach(uid, gid, path, need, table=None):
    """4 read, 2 write, 1 search: every folder above the path inside its volume needs search."""
    if uid == 0 or owner_key(path) is None:
        return True
    table = load_owners() if table is None else table
    path = Path(os.path.abspath(path))
    top = ROOT / "volumes" / owner_key(path).split("/")[0]
    above = [p for p in reversed(path.parents) if p == top or top in p.parents]
    if not path.exists() and need & 2 and path.parent.exists():
        return reach(uid, gid, path.parent, 3, table)
    if not all(bits_for(uid, gid, p, table) & 1 for p in above):
        return False
    return not path.exists() or bits_for(uid, gid, path, table) & need == need


def tool_chown(argv):
    with open(ROOT / "chown.log", "a") as handle:
        handle.write(" ".join(argv) + "\n")
    args = list(argv)
    recursive = args[:1] == ["-R"]
    if recursive:
        args = args[1:]
    uid_text, _, gid_text = args[0].partition(":")
    table = load_owners()
    for path in args[1:]:
        for target in (tree_of(path) if recursive else [path]):
            if owner_key(target):
                table[owner_key(target)] = [int(uid_text), int(gid_text)]
    save_owners(table)
    return 0


def load():
    return json.loads(STATE.read_text())


def save(state):
    tmp = STATE.with_name("state.json.tmp")
    tmp.write_text(json.dumps(state))
    os.replace(tmp, STATE)
    write_view(state, ROOT)


def log(**record):
    with open(LOG, "a") as handle:
        handle.write(json.dumps(record) + "\n")


def start_record(tool, argv, stdin_bytes=None):
    seq = "%d-%d" % (time.time_ns(), os.getpid())
    record = {"seq": seq, "tool": tool, "argv": argv}
    if stdin_bytes is not None:
        (ROOT / "stdin").mkdir(exist_ok=True)
        path = ROOT / "stdin" / (seq + ".bin")
        path.write_bytes(stdin_bytes)
        record["stdin_file"] = str(path)
    log(**record)
    return seq


def end_record(seq, rc, out=b"", err=b""):
    log(seq=seq, end=True, rc=rc, stdout=out.decode("utf-8", "replace"),
        stderr=err.decode("utf-8", "replace"))


def emit(out=b"", err=b""):
    sys.stdout.buffer.write(out)
    sys.stdout.buffer.flush()
    sys.stderr.buffer.write(err)
    sys.stderr.buffer.flush()


def vol_dir(name):
    return ROOT / "volumes" / name


def mount_pairs(container):
    pairs = [(m["dest"], str(vol_dir(m["name"]))) for m in container.get("mounts", [])
             if m.get("type") == "volume"]
    return sorted(pairs, key=lambda pair: -len(pair[0]))


def map_text(text, pairs):
    for dest, host in pairs:
        text = re.sub(r"(?<![\w./-])" + re.escape(dest) + r"(?![\w.-])", host, text)
    return text


def tool_hostname(argv):
    seq = start_record("hostname", argv)
    out = (load()["hostname"] + "\n").encode()
    emit(out)
    end_record(seq, 0, out)
    return 0


def tool_sleep(argv):
    seq = start_record("sleep", argv)
    end_record(seq, 0)
    return 0


CURL_VALUE_FLAGS = {"-K", "--config", "-o", "--output", "-w", "--write-out", "--max-redirs",
                    "--max-time", "-m", "--connect-timeout", "-H", "--header", "-X", "--request"}


def tool_curl(argv):
    options, positional, index = {}, [], 0
    while index < len(argv):
        arg = argv[index]
        if arg in CURL_VALUE_FLAGS and index + 1 < len(argv):
            options[arg] = argv[index + 1]
            index += 2
            continue
        if not arg.startswith("-"):
            positional.append(arg)
        index += 1
    wants_stdin = options.get("-K") == "-" or options.get("--config") == "-"
    stdin_bytes = sys.stdin.buffer.read() if wants_stdin else None
    seq = start_record("curl", argv, stdin_bytes)
    spec = load()["curl"]
    code, redirect, rc = spec["code"], spec.get("redirect", ""), spec.get("rc", 0)
    out = b""
    write_out = options.get("-w") or options.get("--write-out")
    if write_out:
        shown = 0 if rc else code
        text = write_out.replace("%{http_code}", "%03d" % shown)
        text = text.replace("%{redirect_url}", "" if rc else redirect)
        out = text.encode()
    emit(out)
    end_record(seq, rc, out)
    return rc
'''

# The only docker --format shapes the fake understands; the scripts must use exactly these.
F_IMAGE = "{{.Image}}"
F_ENV_NAMES = '{{range .Config.Env}}{{println (index (split . "=") 0)}}{{end}}'
F_MOUNT_TYPES = "{{range .Mounts}}{{println .Type .Name}}{{end}}"
F_MOUNT_DEST = "{{range .Mounts}}{{println .Name .Destination}}{{end}}"
F_STATE = "{{.State.Running}} {{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}"
F_HERMES_HOME = ('{{range .Config.Env}}{{$p := split . "="}}{{if eq (index $p 0) "HERMES_HOME"}}'
                 '{{index $p 1}}{{end}}{{end}}')
F_HEALTH_KIND = "{{if .Config.Healthcheck}}{{index .Config.Healthcheck.Test 0}}{{else}}NONE{{end}}"
# The owner app's configured user, then each token file setting, one Go-quoted line each (no other setting is read).
TOKEN_VARS = [*SURFACE_ENV.values(), *KEY_ENV.values()]
F_APP_SETTINGS = ('{{println (printf "%q" .Config.User)}}{{range .Config.Env}}{{$p := split . "="}}'
                  '{{if eq (index $p 0) ' + " ".join(f'"{name}"' for name in TOKEN_VARS) + '}}'
                  '{{println (printf "%q" .)}}{{end}}{{end}}')
ID_SCRIPT = "id -u && id -g"  # the probe's shell text: a failed first read must stop the second

FAKE_DOCKER = r'''

def finish(seq, rc, out=b"", err=b""):
    emit(out, err)
    end_record(seq, rc, out, err)
    return rc


def refuse(argv, why):
    seq = start_record("docker", argv)
    return finish(seq, 98, err=("fake docker: %s\n" % why).encode())


def tool_docker(argv):
    if not argv:
        return refuse(argv, "no subcommand")
    sub, rest = argv[0], argv[1:]
    if sub == "inspect":
        return d_inspect(argv, rest)
    if sub in ("start", "restart", "stop"):
        return d_lifecycle(argv, sub, rest)
    if sub == "exec":
        return d_exec(argv, rest)
    if sub == "run":
        return d_run(argv, rest)
    return refuse(argv, "subcommand not allowed: " + sub)


def go_quote(text):
    """Go's %q: one double-quoted line; a quote, a backslash or a control character becomes an escape."""
    named = {"\a": "\\a", "\b": "\\b", "\f": "\\f", "\n": "\\n", "\r": "\\r", "\t": "\\t", "\v": "\\v",
             '"': '\\"', "\\": "\\\\"}
    return '"' + "".join(named.get(ch) or (ch if ch.isprintable() else "\\x%02x" % ord(ch)) for ch in text) + '"'


def config_user(entry):
    """Config.User: the text the container was created with (a name or numbers), else its modelled ids."""
    if "config_user" in entry:
        return entry["config_user"]
    return "%d:%d" % tuple(entry.get("user", [0, 0]))


def d_inspect(argv, args):
    seq = start_record("docker", argv)
    kind = fmt = None
    names, i = [], 0
    while i < len(args):
        if args[i] == "--type" and i + 1 < len(args):
            kind, i = args[i + 1], i + 2
        elif args[i] in ("--format", "-f") and i + 1 < len(args):
            fmt, i = args[i + 1], i + 2
        else:
            names.append(args[i])
            i += 1
    if kind not in ("container", "image") or fmt is None or len(names) != 1:
        return finish(seq, 97, err=b"fake docker: inspect needs --type, --format and one name\n")
    state = load()
    table = state["containers"] if kind == "container" else state["images"]
    entry = table.get(names[0])
    if entry is None:
        return finish(seq, 1, err=("Error: No such %s: %s\n" % (kind, names[0])).encode())
    if fmt == F_IMAGE and kind == "container":
        out = entry["image"] + "\n"
    elif fmt == F_ENV_NAMES:
        out = "".join(item.split("=", 1)[0] + "\n" for item in entry["env"])
    elif fmt == F_MOUNT_TYPES and kind == "container":
        out = "".join("%s %s\n" % (m["type"], m.get("name", "")) for m in entry["mounts"])
    elif fmt == F_MOUNT_DEST and kind == "container":
        out = "".join("%s %s\n" % (m.get("name", ""), m["dest"]) for m in entry["mounts"])
    elif fmt == F_HERMES_HOME and kind == "container":
        out = "".join(item.partition("=")[2] for item in entry["env"] if item.startswith("HERMES_HOME=")) + "\n"
    elif fmt == F_APP_SETTINGS and kind == "container":  # read from the configuration: the container need not run
        lines = [go_quote(config_user(entry))]
        lines += [go_quote(item) for item in entry["env"] if item.partition("=")[0] in TOKEN_VARS]
        out = "\n".join(lines) + "\n"
    elif fmt == F_HEALTH_KIND and kind == "container":
        test = entry.get("healthcheck")  # the configured Healthcheck.Test: absent, empty or [kind, ...]
        if test is not None and not test:
            return finish(seq, 1, err=b"template: index out of range: 0\n")
        out = (test[0] if test else "NONE") + "\n"
    elif fmt == F_STATE and kind == "container":
        health = entry.get("health") or "none"
        if entry.get("starting_polls", 0) > 0 and entry.get("health"):
            entry["starting_polls"] -= 1
            health = "starting"
            save(state)
        out = "%s %s\n" % ("true" if entry["running"] else "false", health)
    else:
        return finish(seq, 97, err=b"fake docker: format not supported by the fake\n")
    return finish(seq, 0, (out + "\n").encode())


BOOT_SUBS = ["cron", "sessions", "logs", "hooks", "memories", "skills", "skins", "plans", "workspace",
             "home", "profiles", "pairing", "platforms/pairing", "lazy-packages"]


def boot_ownership(entry):
    """The image's boot hook (docker/stage2-hook.sh): at every start the home goes to the runtime account."""
    runtime = entry.get("runtime")
    value = dict(item.partition("=")[::2] for item in entry["env"]).get("HERMES_HOME")
    if not runtime or not value:
        return
    home = Path(map_text(value, mount_pairs(entry)))
    if owner_key(home) is None or not home.is_dir():
        return
    table, want = load_owners(), list(runtime)

    def differs(path):
        return any(owner_in(table, item) != want for item in tree_of(path))

    def give(path):
        for item in tree_of(path):
            table[owner_key(item)] = want

    if owner_in(table, home)[0] != want[0]:
        table[owner_key(home)] = want
        for sub in BOOT_SUBS:
            if (home / sub).exists() and differs(home / sub):
                give(home / sub)
    if (home / "profiles").is_dir() and differs(home / "profiles"):
        give(home / "profiles")
    for name in (".env", "config.yaml"):
        if (home / name).is_file() and not (home / name).is_symlink():
            table[owner_key(home / name)] = want
    save_owners(table)


def token_files_readable(entry):
    """What the owner app checks when it starts: every token file its settings name exists and its user reads it."""
    uid, gid = entry.get("user", [0, 0])
    pairs, table = mount_pairs(entry), load_owners()
    values = dict(item.partition("=")[::2] for item in entry["env"])
    for name in TOKEN_VARS:
        path = Path(map_text(values.get(name, ""), pairs))
        if not values.get(name) or owner_key(path) is None or not path.is_file() or not reach(uid, gid, path, 4, table):
            return False
    return True


def d_lifecycle(argv, sub, args):
    seq = start_record("docker", argv)
    if len(args) != 1:
        return finish(seq, 97, err=b"fake docker: one container name expected\n")
    state = load()
    entry = state["containers"].get(args[0])
    if entry is None:
        return finish(seq, 1, err=("Error: No such container: %s\n" % args[0]).encode())
    if sub == "stop":
        entry["running"] = False
        save(state)
        return finish(seq, 0, (args[0] + "\n").encode())
    if entry.get("fails_to_start"):
        return finish(seq, 1, err=b"Error: cannot start container\n")
    entry["running"] = not entry.get("stays_down")
    if entry.get("exits_without_token_files") and not token_files_readable(entry):
        entry["running"] = False  # docker start still succeeds; the app is gone a moment later
    boot_ownership(entry)
    if entry["running"] and entry.get("health_on_start"):  # a container that never ran has no health state yet
        entry["health"] = entry["health_on_start"]
        entry["starting_polls"] = entry.get("health_delay", 0)
    elif sub == "restart" and entry.get("health"):
        entry["starting_polls"] = entry.get("health_delay", 0)
    save(state)
    return finish(seq, 0, (args[0] + "\n").encode())


def unmap(data, pairs):
    """Show the container's view: host folders in the output become the mount destinations."""
    text = data.decode("utf-8", "replace")
    for dest, host in pairs:
        text = text.replace(host, dest)
    return text.encode()


def container_env(entry, pairs, uid=0, gid=0):
    env = {"FAKE_EXEC_UID": str(uid), "FAKE_EXEC_GID": str(gid), "PATH": "%s:%s:%s:/usr/bin:/bin" % (ROOT / "ctr" / "shim", ROOT / "ctr" / "venv",
                                               ROOT / "ctr" / "bin"),
           # The stand-in modules come first; hermes_cli then also reaches the platform's own modules (its
           # __init__ extends the package path), whose bytecode is read from the shared cache.
           "FAKE_HOST_ROOT": str(ROOT), "PYTHONPATH": "%s:%s" % (ROOT / "ctr" / "pkgs", REPO_ROOT),
           "PYTHONPYCACHEPREFIX": REAL_PYC, "PYTHONDONTWRITEBYTECODE": "1", "HOME": str(ROOT / "ctr" / "home")}
    for item in entry["env"]:
        key, _, value = item.partition("=")
        if key != "PATH":
            env[key] = map_text(value, pairs)
    return env


def parse_user(text):
    uid_text, _, gid_text = text.partition(":")
    if not uid_text.isdigit() or (gid_text and not gid_text.isdigit()):
        return None
    return int(uid_text), int(gid_text or 0)  # docker: an id without a passwd entry gets group 0


def redirect_targets(local):
    """Paths that a `sh -c` script writes with > or >> to "$N" (N counts from the first argument)."""
    if local[:2] != ["sh", "-c"] or len(local) < 3:
        return []
    return [local[3 + int(n)] for n in re.findall(r'>>?\s*"\$(\d)"', local[2]) if 3 + int(n) < len(local)]


def is_anchor_call(container, interactive, user, cmd, stdin_bytes):
    """The reset's Project call or its board write, whole: container, account, words and (for the board) input."""
    if container != DASH_C or user is None:
        return False
    if interactive:
        return cmd == BOARD_WRITE and stdin_bytes == BOARD_PROGRAM.encode()
    return cmd == PROJECT_CREATE


def d_exec(argv, args):
    import subprocess
    interactive, user, i = False, None, 0
    while i < len(args) and args[i].startswith("-"):
        if args[i] in ("-i", "--interactive"):
            interactive, i = True, i + 1
        elif args[i] in ("-u", "--user") and i + 1 < len(args):
            user, i = args[i + 1], i + 2
        else:
            return refuse(argv, "exec flag not supported: " + args[i])
    if i >= len(args) - 1:
        return refuse(argv, "exec needs a container and a command")
    container, cmd = args[i], args[i + 1:]
    stdin_bytes = sys.stdin.buffer.read() if interactive else None
    seq = start_record("docker", argv, stdin_bytes)
    entry = load()["containers"].get(container)
    if entry is None:
        return finish(seq, 1, err=("Error: No such container: %s\n" % container).encode())
    if not entry["running"]:
        return finish(seq, 1, err=b"Error response from daemon: container is not running\n")
    if cmd[0] == "printenv" and len(cmd) == 2:
        value = dict(item.partition("=")[::2] for item in entry["env"]).get(cmd[1])
        return finish(seq, 1) if value is None else finish(seq, 0, (value + "\n").encode())
    who = parse_user(user) if user is not None else tuple(entry.get("user", [0, 0]))
    if who is None:
        return finish(seq, 126, err=b"Error response from daemon: unable to find user: only numbers are modelled\n")
    uid, gid = who
    pairs = mount_pairs(entry)
    local = [map_text(part, pairs) for part in cmd]
    said = entry.get("answers", {}).get(" ".join(cmd))
    if said is not None:
        return finish(seq, 0, said.encode() + (b"\n" if said else b""))
    if cmd in (["id", "-u"], ["id", "-g"]):
        return finish(seq, 0, ("%d\n" % (uid if cmd[1] == "-u" else gid)).encode())
    if cmd[:3] == ["stat", "-c", "%u:%g"] and len(cmd) == 4:
        if not Path(local[3]).exists():
            return finish(seq, 1, err=("stat: cannot statx '%s': No such file or directory\n" % cmd[3]).encode())
        return finish(seq, 0, ("%d:%d\n" % tuple(owner_in(load_owners(), local[3]))).encode())
    if is_anchor_call(container, interactive, user, cmd, stdin_bytes) and not load().get("hermes", {}).get("real_platform"):
        return finish(seq, 0)  # recorded whole above; the platform's code runs only where a test reads what it wrote
    targets = redirect_targets(local)
    for path in targets:
        if not reach(uid, gid, path, 2):
            return finish(seq, 2, err=("sh: 1: cannot create %s: Permission denied\n" % cmd[local.index(path)]).encode())
    volumes = [host for _, host in pairs]
    known, owned_before = {path for host in volumes for path in tree_of(host)}, load_owners()
    if local[0] in ("python3", "python"):
        local[0] = sys.executable
    try:
        done = subprocess.run(local, input=stdin_bytes or b"", capture_output=True,
                              env=container_env(entry, pairs, uid, gid), cwd=str(ROOT / "ctr" / "home"))
    except FileNotFoundError:
        return finish(seq, 127, err=b"exec: command not found\n")
    table = load_owners()
    # What the command created inside a volume belongs to the identity it ran as, unless it named an owner itself.
    for host in volumes:
        for path in tree_of(host):
            key = owner_key(path)
            if path not in known and key and table.get(key) == owned_before.get(key):
                table[key] = [uid, gid]
    save_owners(table)
    return finish(seq, done.returncode, unmap(done.stdout, pairs), unmap(done.stderr, pairs))


def d_run_id(seq, state, image, opts, flags, command):
    """The read-only helper (no network, no volume): a real shell runs the script text with a stand-in `id`."""
    import shlex
    import subprocess
    text = command[1] if command[:1] == ["-c"] and len(command) == 2 else ""
    if (flags != {"--rm"} or opts.get("--network") != "none" or opts.get("--entrypoint") != "sh" or "--user" not in opts
            or not re.fullmatch(r"id -[ug](?: *(?:&&|\|\||;) *id -[ug])*", text)):
        return finish(seq, 97, err=b"fake docker: run shape not supported\n")
    spec, users = opts["--user"], state["images"][image].get("users", {})
    if spec == "":
        who = (0, 0)
    elif parse_user(spec) is not None:
        who = parse_user(spec)
    elif spec in users:
        who = tuple(users[spec])
    else:
        return finish(seq, 126, err=("docker: unable to find user %s: no matching entries in passwd file\n" % spec).encode())
    said, codes = {}, {}
    for box in state["containers"].values():  # what a container of this image is made to answer to id
        if box["image"] == image:
            said.update(box.get("answers", {}))
            codes.update(box.get("exit_codes", {}))
    reads = ""
    for flag, number in (("-u", who[0]), ("-g", who[1])):
        out = said.get("id " + flag, str(number))
        show = "" if out is None else "printf '%s\\n' " + shlex.quote(out) + "; "
        reads += "%s) %sreturn %d ;; " % (flag, show, codes.get("id " + flag, 0))
    shell = "id() { case $1 in %s*) return 127 ;; esac; }\n%s" % (reads, text)
    done = subprocess.run(["sh", "-c", shell], input=b"", capture_output=True, env={"PATH": "/usr/bin:/bin"})
    return finish(seq, done.returncode, done.stdout, done.stderr)


def d_run(argv, args):
    import subprocess
    opts, flags, i = {}, set(), 0
    while i < len(args) and args[i].startswith("-"):
        if args[i] in ("--rm", "-i"):
            flags.add(args[i])
            i += 1
        elif args[i] in ("--network", "--user", "--entrypoint", "--mount") and i + 1 < len(args):
            opts[args[i]], i = args[i + 1], i + 2
        else:
            return refuse(argv, "run flag not supported: " + args[i])
    if i >= len(args):
        return refuse(argv, "run needs an image")
    image, command = args[i], args[i + 1:]
    stdin_bytes = sys.stdin.buffer.read() if "-i" in flags else None
    seq = start_record("docker", argv, stdin_bytes)
    state = load()
    if image not in state["images"]:
        return finish(seq, 125, err=b"Unable to find image locally; no network\n")
    if "--mount" not in opts:
        return d_run_id(seq, state, image, opts, flags, command)
    mount = dict(part.partition("=")[::2] for part in opts.get("--mount", "").split(","))
    source, target = mount.get("source"), mount.get("target")
    if opts.get("--entrypoint") != "sh" or command[:1] != ["-c"] or len(command) != 2 or not source:
        return finish(seq, 97, err=b"fake docker: run shape not supported\n")
    if source not in state["volumes"]:
        state["volumes"].append(source)
        state.setdefault("created_volumes", []).append(source)
        save(state)
    host = vol_dir(source)
    host.mkdir(parents=True, exist_ok=True)
    script = re.sub(r"(?<![\w./-])" + re.escape(target) + r"(?![\w.-])", str(host), command[1])
    if re.search(r"(?<![\w$.-])/(?!dev/)[A-Za-z]", script.replace(str(host), "")):
        return finish(seq, 99, err=b"fake docker: script names a path outside the volume\n")
    env = {"PATH": "%s:/usr/bin:/bin" % (ROOT / "ctr" / "bin"), "FAKE_HOST_ROOT": str(ROOT)}
    done = subprocess.run(["sh", "-c", script], input=stdin_bytes or b"", capture_output=True,
                          env=env, cwd=str(ROOT / "ctr" / "home"))
    return finish(seq, done.returncode, done.stdout, done.stderr)
'''

FAKE_PLATFORM = r'''

import stat

EMPTY_CRON = ("No scheduled jobs.\n"
              "Create one with 'hermes cron create ...' or the /cron command in chat.\n")
SURFACES = {"workspace", "recommendations", "connections", "automations", "models"}


def refused(message):
    print("error: " + message)
    return 2


def issue_token(args):
    """The checks of token_store's plaintext output writer and of the audit log, for the exec identity."""
    opts, i = {}, 0
    while i < len(args):
        if args[i] in ("--surface", "--out", "--ttl-days", "--ttl-hours") and i + 1 < len(args):
            opts[args[i]], i = args[i + 1], i + 2
        else:
            sys.stderr.write("fake hermes: bad option\n")
            return 2
    surface, out = opts.get("--surface"), opts.get("--out")
    if surface not in SURFACES or not out:
        return 2
    uid, gid = int(os.environ.get("FAKE_EXEC_UID", "0")), int(os.environ.get("FAKE_EXEC_GID", "0"))
    table = load_owners()
    logs = Path(os.environ["HERMES_HOME"]) / "logs"
    if not reach(uid, gid, logs if logs.exists() else logs.parent, 2 if logs.exists() else 3, table):
        return refused("dashboard-auth audit log write failed: [Errno 13] Permission denied: %s" % logs)
    if load().get("hermes", {}).get("issue_fail") == surface:
        return refused("issue failed")
    path = Path(out)
    parent = path.parent
    if parent.is_symlink() or not parent.is_dir():
        return refused("parent directory %s must be an existing non-symlink directory" % parent)
    if not reach(uid, gid, parent, 0, table):
        return refused("parent directory %s cannot be read: [Errno 13] Permission denied" % parent)
    if owner_in(table, parent)[0] != uid:
        return refused("parent directory %s is not owned by the current user" % parent)
    mode = stat.S_IMODE(parent.stat().st_mode)
    if mode != 0o700:
        return refused("parent directory %s is not owner-only (mode 0o%o) - mode must be exactly 0o700"
                       % (parent, mode))
    if path.exists() or path.is_symlink():
        return refused("%s already exists - refusing to overwrite it" % path)
    fd = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "w") as handle:
        handle.write("tok-%s-%s" % (surface, os.urandom(12).hex()))
    os.chmod(out, 0o600)
    table[owner_key(path)] = [uid, gid]
    save_owners(table)
    print("issued surface=%s" % surface)
    return 0


CRON_LIST_PROGRAM = "import sys\nfrom hermes_cli.cron import cron_list\ncron_list(sys.argv[1] == '1')\n"
# `hermes project ...`: the platform's own parser and handler for the project commands.
PROJECT_PROGRAM = (
    "import argparse, sys\n"
    "from hermes_cli import projects_cmd\n"
    "top = argparse.ArgumentParser(prog='hermes')\n"
    "projects_cmd.build_parser(top.add_subparsers(dest='command'))\n"
    "sys.exit(projects_cmd.projects_command(top.parse_args(sys.argv[1:])))\n")


def real_platform(program, *argv):
    """Runs a program on the platform's own modules: the repository on PYTHONPATH, this folder as the home."""
    import subprocess
    env = {"PATH": os.environ["PATH"], "HOME": os.environ["HOME"], "HERMES_HOME": os.environ["HERMES_HOME"],
           "PYTHONPATH": REPO_ROOT, "PYTHONPYCACHEPREFIX": REAL_PYC, "AWS_EC2_METADATA_DISABLED": "true"}
    return subprocess.run([sys.executable, "-c", program, *argv], env=env, capture_output=True, text=True)


def hermes_main(args):
    spec = load().get("hermes", {})
    home = Path(os.environ["HERMES_HOME"])
    if args[:2] == ["cron", "list"] and spec.get("real_platform"):
        done = real_platform(CRON_LIST_PROGRAM, "1" if "--all" in args[2:] else "0")
        sys.stdout.write(done.stdout)
        return done.returncode
    if args[:2] == ["cron", "list"]:
        profile = "default" if home.parent.name != "profiles" else (
            {"raphael-planner": "planner", "raphael-designer": "designer"}.get(home.name, home.name))
        jobs = spec.get("cron_jobs", {}).get(profile, [])
        sys.stdout.write(EMPTY_CRON if not jobs else "".join("%s\n" % job for job in jobs))
        return spec.get("cron_rc", 0)
    if args[:1] == ["project"]:
        if spec.get("project_fail") == "refused":
            sys.stderr.write("fake hermes: project command failed\n")
            return 2
        if spec.get("project_fail") == "silent":  # the command answers success and makes nothing
            return 0
        done = real_platform(PROJECT_PROGRAM, *args)
        sys.stdout.write(done.stdout)
        sys.stderr.write(done.stderr)
        return done.returncode
    if args[:1] == ["send"]:
        if "--to" not in args or args[-1].startswith("-"):
            sys.stderr.write("usage: hermes send --to PLATFORM TEXT\n")
            return 2
        rc = spec.get("send_rc", 1)
        if rc:
            sys.stderr.write("hermes send: delivery failed\n")
        return rc
    if args[:2] == ["kanban-workspace-token", "issue"]:
        return issue_token(args[2:])
    sys.stderr.write("fake hermes: unsupported command\n")
    return 2


def main():
    tools = {"docker": tool_docker, "hostname": tool_hostname, "sleep": tool_sleep,
             "curl": tool_curl, "chown": tool_chown}
    sys.exit(tools[sys.argv[1]](sys.argv[2:]))


if __name__ == "__main__":
    main()
'''

FAKE_TOOLS_SOURCE = (
    FAKE_COMMON
    + VIEW_SOURCE
    + "".join("%s = %r\n" % (n, globals()[n])
              for n in ("F_IMAGE", "F_ENV_NAMES", "F_MOUNT_TYPES", "F_MOUNT_DEST", "F_STATE",
                        "F_HERMES_HOME", "F_HEALTH_KIND", "F_APP_SETTINGS", "TOKEN_VARS",
                        "DASH_C", "PROJECT_CREATE", "BOARD_WRITE", "BOARD_PROGRAM"))
    + "REPO_ROOT = %r\nREAL_PYC = %r\n" % (str(REPO), str(REAL_PYC))
    + FAKE_DOCKER
    + FAKE_PLATFORM
)

# Stand-ins for the platform modules the scripts import inside the container.
FAKE_PACKAGES = {
    "hermes_cli/__init__.py": "from pkgutil import extend_path\n__path__ = extend_path(__path__, __name__)\n",
    "hermes_cli/config.py": '''
import os
from pathlib import Path
import yaml


def get_hermes_home():
    return Path(os.environ["HERMES_HOME"])


def load_config():
    config = {"kanban": {"dispatch_in_gateway": True}}
    path = get_hermes_home() / "config.yaml"
    if path.exists():
        user = yaml.safe_load(path.read_text()) or {}
        config["kanban"].update(user.get("kanban") or {})
        config.update({k: v for k, v in user.items() if k != "kanban"})
    return config
''',
    "hermes_cli/profiles.py": '''
import json, os, subprocess, sys
from pathlib import Path

PROGRAM = ("import json, sys\\nfrom hermes_cli.profiles import profiles_to_serve\\n"
           "print(json.dumps([[n, str(p)] for n, p in profiles_to_serve(True, json.loads(sys.argv[1]))]))\\n")


def profiles_to_serve(multiplex, profile_allowlist=None):
    """The platform's own enumeration, run on its real modules with this home."""
    state = json.loads((Path(os.environ["FAKE_HOST_ROOT"]) / "state.json").read_text())
    if state.get("hermes", {}).get("serve_fail"):
        raise RuntimeError("the enumeration cannot be read")
    env = {"PATH": os.environ["PATH"], "HOME": os.environ["HOME"], "HERMES_HOME": os.environ["HERMES_HOME"],
           "PYTHONPATH": @REPO@, "PYTHONPYCACHEPREFIX": @PYC@}
    done = subprocess.run([sys.executable, "-c", PROGRAM, json.dumps(profile_allowlist)], env=env,
                          capture_output=True, text=True)
    if done.returncode:
        raise RuntimeError("the enumeration failed")
    served = [(name, Path(home)) for name, home in json.loads(done.stdout)]
    return served + [(name, Path(os.environ["HERMES_HOME"]) / "profiles" / name)
                     for name in state.get("hermes", {}).get("serve_extra", [])]
'''.replace("@REPO@", repr(str(REPO))).replace("@PYC@", repr(str(REAL_PYC))),
    "hermes_cli/send_cmd.py": '''
import os
from pathlib import Path


def _load_hermes_env():
    path = Path(os.environ["HERMES_HOME"]) / ".env"
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip() and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ[key] = value
''',
    "gateway/__init__.py": "",
    "gateway/config.py": '''
import os
from pathlib import Path
import yaml

ENV_PLATFORMS = {"TELEGRAM_BOT_TOKEN": "telegram", "DISCORD_BOT_TOKEN": "discord",
                 "SLACK_BOT_TOKEN": "slack", "API_SERVER_KEY": "api_server"}


class Platform:
    def __init__(self, value):
        self.value = value


class GatewayConfig:
    def get_connected_platforms(self):
        names = {v for k, v in ENV_PLATFORMS.items() if os.environ.get(k)}
        path = Path(os.environ["HERMES_HOME"]) / "config.yaml"
        if path.exists():
            platforms = (yaml.safe_load(path.read_text()) or {}).get("platforms") or {}
            names |= {n for n, block in platforms.items() if (block or {}).get("enabled")}
        return [Platform(n) for n in sorted(names)]


def load_gateway_config():
    return GatewayConfig()
''',
}

_KEY_PEM: list[bytes] = []


def signing_key_pem() -> bytes:
    """One RSA key for the fake proxy volume, made once per test run."""
    if not _KEY_PEM:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric import rsa

        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        _KEY_PEM.append(key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        ))
    return _KEY_PEM[0]


def is_id_probe(argv: list[str]) -> bool:
    """The one read-only helper run: no volume, no input, only id printed (it changes nothing)."""
    return argv[:1] == ["run"] and argv[-2:] == ["-c", ID_SCRIPT] and "--mount" not in argv and "-i" not in argv


def docker_target(call: dict) -> tuple[str, str | None]:
    """Subcommand and container name of a recorded docker call (None when it names none)."""
    argv = call["argv"]
    sub = argv[0]
    if sub in ("start", "restart", "stop"):
        return sub, argv[1]
    if sub == "inspect":
        return sub, argv[-1]
    if is_id_probe(argv):
        return "id-run", None
    if sub == "exec":
        i = 1
        while argv[i].startswith("-"):
            i += 2 if argv[i] in ("-u", "--user") else 1
        return sub, argv[i]
    return sub, None


def docker_names(call: dict) -> set[str]:
    """Container and volume style names in a recorded docker call. The anchor slug is spelled like the production
    container's name; inside the command that an exec runs it is the platform's own data, not a docker resource."""
    argv = call["argv"]
    sub, container = docker_target(call)
    first = argv.index(container) + 1 if sub == "exec" else len(argv)
    shown = argv[:first] + [part for part in argv[first:] if part != ANCHOR]
    return set().union(*(resource_names_in(part) for part in shown))


FAKE_LIB_SH = r'''
jesc() {
  REPLY=${1//\\/\\\\}
  REPLY=${REPLY//\"/\\\"}
  REPLY=${REPLY//$'\n'/\\n}
  REPLY=${REPLY//$'\t'/\\t}
  REPLY=${REPLY//$'\r'/\\r}
}

plain() {
  local t=${1//[$'\n\t\r']/}
  [[ $t != *[[:cntrl:]]* ]]
}

# log_call TOOL RC STDOUT STDERR ARGV...: the same two JSON lines the Python fake writes
log_call() {
  local tool=$1 rc=$2 out=$3 err=$4 seq a args=""
  shift 4
  seq=${EPOCHREALTIME:-$SECONDS.$RANDOM}-$$
  for a; do
    jesc "$a"
    args+="${args:+, }\"$REPLY\""
  done
  jesc "$out"
  out=$REPLY
  jesc "$err"
  printf '{"seq": "%s", "tool": "%s", "argv": [%s]}\n{"seq": "%s", "end": true, "rc": %d, "stdout": "%s", "stderr": "%s"}\n' \
    "$seq" "$tool" "$args" "$seq" "$rc" "$out" "$REPLY" >> "$FAKE_HOST_ROOT/calls.jsonl"
}
'''

HOSTNAME_SH = '''#!/bin/bash
. "$FAKE_HOST_ROOT/fake_lib.sh"
. "$FAKE_HOST_ROOT/state.sh"
printf '%s\\n' "$HOST"
log_call hostname 0 "$HOST"$'\\n' "" "$@"
'''

SLEEP_SH = '''#!/bin/bash
. "$FAKE_HOST_ROOT/fake_lib.sh"
log_call sleep 0 "" "" "$@"
'''

# The real curl behind a wrapper that only records the argument list (the start line of a call, without an
# end line); it execs curl, so stdin, stdout and the exit status are curl's own.
REAL_CURL_SH = r'''#!/bin/bash
. "$FAKE_HOST_ROOT/fake_lib.sh"
args=""
for a; do
  jesc "$a"
  args+="${args:+, }\"$REPLY\""
done
printf '{"seq": "%s", "tool": "curl", "argv": [%s]}\n' "${EPOCHREALTIME:-$SECONDS.$RANDOM}-$$" "$args" >> "$FAKE_HOST_ROOT/calls.jsonl"
exec @CURL@ "$@"
'''

# Read-only docker calls of exactly the shapes the scripts use run here; anything else goes to the
# Python fake, which keeps the full behaviour (including its refusals).
DOCKER_SH = r'''#!/bin/bash
. "$FAKE_HOST_ROOT/fake_lib.sh"
slow() { exec @PYEXE@ -S -E "$FAKE_HOST_ROOT/fake_tools.py" docker "$@"; }
F_IMAGE=@F_IMAGE@
F_ENV_NAMES=@F_ENV_NAMES@
F_MOUNT_TYPES=@F_MOUNT_TYPES@
F_MOUNT_DEST=@F_MOUNT_DEST@
F_STATE=@F_STATE@
F_HERMES_HOME=@F_HERMES_HOME@
[ -f "$FAKE_HOST_ROOT/state.sh" ] || slow "$@"
. "$FAKE_HOST_ROOT/state.sh"
for arg; do plain "$arg" || slow "$@"; done
name_ok='^[A-Za-z0-9][A-Za-z0-9_.:@-]*$'
out="" err="" rc=0
if [ "$#" -eq 6 ] && [ "$1" = inspect ] && [ "$2" = --type ] && [ "$4" = --format ] && [[ $6 =~ $name_ok ]]; then
  kind=$3 fmt=$5 name=$6
  if [ "$kind" = container ]; then
    case $fmt in "$F_IMAGE" | "$F_ENV_NAMES" | "$F_MOUNT_TYPES" | "$F_MOUNT_DEST" | "$F_STATE" | "$F_HERMES_HOME") ;; *) slow "$@" ;; esac
    if [[ ! -v IMG[$name] ]]; then
      rc=1 err="Error: No such container: $name"$'\n'
    else
      case $fmt in
        "$F_IMAGE") out=${IMG[$name]}$'\n' ;;
        "$F_ENV_NAMES") out=${ENVN[$name]} ;;
        "$F_MOUNT_TYPES") out=${MTYPES[$name]} ;;
        "$F_MOUNT_DEST") out=${MDEST[$name]} ;;
        "$F_HERMES_HOME")
          value=""
          while IFS= read -r line; do
            if [[ $line == HERMES_HOME=* ]]; then value+=${line#*=}; fi
          done <<<"${ENVL[$name]}"
          out=$value$'\n' ;;
        "$F_STATE")
          health=${HEALTH[$name]} polls=${POLLS[$name]}
          if [ -f "$FAKE_HOST_ROOT/polls/$name" ]; then read -r polls < "$FAKE_HOST_ROOT/polls/$name"; fi
          if [ -n "$health" ] && [ "$polls" -gt 0 ]; then
            mkdir -p "$FAKE_HOST_ROOT/polls"
            echo $((polls - 1)) > "$FAKE_HOST_ROOT/polls/$name"
            health=starting
          fi
          running=false
          if [ "${RUN[$name]}" = 1 ]; then running=true; fi
          out="$running ${health:-none}"$'\n'
          ;;
      esac
      out+=$'\n'
    fi
  elif [ "$kind" = image ] && [ "$fmt" = "$F_ENV_NAMES" ]; then
    if [[ ! -v IMGENV[$name] ]]; then
      rc=1 err="Error: No such image: $name"$'\n'
    else
      out=${IMGENV[$name]}$'\n'
    fi
  else
    slow "$@"
  fi
elif [ "$#" -eq 4 ] && [ "$1" = exec ] && [ "$3" = printenv ] && [[ $2 =~ $name_ok ]] && [[ $4 =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
  name=$2 var=$4
  if [[ ! -v IMG[$name] ]]; then
    rc=1 err="Error: No such container: $name"$'\n'
  elif [ "${RUN[$name]}" != 1 ]; then
    rc=1 err=$'Error response from daemon: container is not running\n'
  else
    found=0 value=""
    while IFS= read -r line; do
      if [[ $line == *=* ]]; then key=${line%%=*} val=${line#*=}; else key=$line val=""; fi
      if [ "$key" = "$var" ]; then found=1 value=$val; fi
    done <<<"${ENVL[$name]}"
    if [ "$found" = 1 ]; then out=$value$'\n'; else rc=1; fi
  fi
else
  slow "$@"
fi
printf '%s' "$out"
printf '%s' "$err" >&2
log_call docker "$rc" "$out" "$err" "$@"
exit "$rc"
'''


class Host:
    """A fake Server 2: state file, volume folders, fake tools and a way to run the scripts."""

    def __init__(self, root: Path, build: bool = True):
        self.root = root
        if build:
            self._build()

    @classmethod
    def copy_of(cls, other: "Host", root: Path) -> "Host":
        """A second host with the same state and volumes, and an empty call log."""
        records = {"calls.jsonl", "chown.log", "stdin", "polls", "interp.log"}
        shutil.copytree(other.root, root, symlinks=True, ignore=lambda folder, names: [
            name for name in names if Path(folder) == other.root and name in records])
        clone = cls(root, build=False)
        clone._write_tools()
        return clone

    @staticmethod
    def _exe(path: Path, text: str) -> None:
        path.write_text(text)
        path.chmod(0o755)

    def _write_tools(self) -> None:
        root = self.root
        (root / "fake_tools.py").write_text(FAKE_TOOLS_SOURCE)
        (root / "fake_lib.sh").write_text(FAKE_LIB_SH)
        quote = _view_namespace["_q"]
        self._exe(root / "bin/curl",
                  f'#!/bin/sh\nexec {quote(sys.executable)} -S -E "$FAKE_HOST_ROOT/fake_tools.py" curl "$@"\n')
        self._exe(root / "bin/hostname", HOSTNAME_SH)
        self._exe(root / "bin/sleep", SLEEP_SH)
        docker = DOCKER_SH.replace("@PYEXE@", quote(sys.executable))
        for name in ("F_IMAGE", "F_ENV_NAMES", "F_MOUNT_TYPES", "F_MOUNT_DEST", "F_STATE", "F_HERMES_HOME"):
            docker = docker.replace(f"@{name}@", quote(globals()[name]))
        self._exe(root / "bin/docker", docker)
        # Like the published image: a /bin/sh shim sits first on PATH and runs the venv binary,
        # whose shebang names the venv's own interpreter (a wrapper here, so the tests see its use).
        self._exe(root / "ctr/shim/hermes", '#!/bin/sh\nexec "$FAKE_HOST_ROOT/ctr/venv/hermes" "$@"\n')
        self._exe(root / "ctr/venv/python",
                  f'#!/bin/sh\nprintf \'%s\\n\' "$1" >> "$FAKE_HOST_ROOT/interp.log"\nexec "{sys.executable}" "$@"\n')
        self._exe(root / "ctr/venv/hermes",
                  f"#!{root}/ctr/venv/python\nimport os, sys\nsys.path.insert(0, os.environ['FAKE_HOST_ROOT'])\n"
                  "import fake_tools\nsys.exit(fake_tools.hermes_main(sys.argv[1:]))\n")
        self._exe(root / "ctr/bin/chown",
                  f'#!/bin/sh\nexec {quote(sys.executable)} -S -E "$FAKE_HOST_ROOT/fake_tools.py" chown "$@"\n')

    def use_real_curl(self, curl: str) -> None:
        """Put the real curl (an absolute path) in place of the fake one, behind the recording wrapper."""
        self._exe(self.root / "bin/curl", REAL_CURL_SH.replace("@CURL@", _view_namespace["_q"](curl)))

    def _build(self) -> None:
        root = self.root
        for sub in ("bin", "ctr/shim", "ctr/venv", "ctr/bin", "ctr/home", "ctr/pkgs", "home", "work"):
            (root / sub).mkdir(parents=True)
        self._write_tools()
        for rel, text in FAKE_PACKAGES.items():
            target = root / "ctr/pkgs" / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text)
        for volume in STACK_VOLUMES + DECOY_VOLUMES:
            (root / "volumes" / volume).mkdir(parents=True)
            (root / "volumes" / volume / "keep.txt").write_text(f"old content of {volume}\n")
        (root / "volumes" / HOME_V / "stale").mkdir()
        (root / "volumes" / HOME_V / "stale/old.db").write_text("old rows\n")
        (root / "volumes" / TOKENS_V / "workspace").write_text("old-token-value\n")
        key = root / "volumes" / PROXY_V / "signing-key.pem"
        key.write_bytes(signing_key_pem())
        key.chmod(0o644)
        state = self._initial_state()
        (root / "state.json").write_text(json.dumps(state))
        write_view(state, root)
        self.set_identities((1000, 1000), (10000, 10000))

    @staticmethod
    def _initial_state() -> dict:
        img = {
            "sha256:img-proxy": ["PATH=/usr/bin", "PYTHON_VERSION=3.12"],
            "sha256:img-app": ["PATH=/usr/bin", "NODE_VERSION=22", "YARN_VERSION=1"],
            "sha256:img-hermes": ["PATH=/opt/hermes/bin:/usr/bin", "HERMES_HOME=/opt/data"],
            "sha256:img-browser": ["PATH=/usr/bin", "PLAYWRIGHT_BROWSERS_PATH=/ms-playwright"],
            "sha256:img-prod": ["PATH=/usr/bin", "NODE_VERSION=22"],
            "sha256:img-old": ["PATH=/usr/bin"],
        }
        values = {name: "placeholder" for name in APP_ENV_NAMES}
        for surface, name in SURFACE_ENV.items():
            values[name] = f"/run/tokens/{surface}"
        values[KEY_ENV["default"]] = "/run/tokens/executor"
        values[KEY_ENV["planner"]] = "/run/tokens/planner"
        app_env = [f"{key}={value}" for key, value in values.items()]

        def vol(name, dest):
            return {"type": "volume", "name": name, "dest": dest}

        checked = {"healthcheck": ["CMD-SHELL", "probe"], "health": "healthy"}

        def box(image, env, mounts, **extra):
            own = {item.partition("=")[0] for item in env}  # like docker: the container's value replaces the image's
            merged = [item for item in img[image] if item.partition("=")[0] not in own] + env
            return {"image": image, "env": merged, "mounts": mounts, "running": True, **extra}

        containers = {
            PROXY_C: box("sha256:img-proxy", ["SYNTHETIC_STATE=/state"], [vol(PROXY_V, "/state")], **checked),
            APP_C: box("sha256:img-app", app_env, [vol(TOKENS_V, "/run/tokens")], user=[1000, 1000], **checked),
            DASH_C: box("sha256:img-hermes", ["HERMES_HOME=/opt/data"],
                        [vol(HOME_V, "/opt/data"), vol(TOKENS_V, "/tokens")],
                        health_delay=1, runtime=[10000, 10000], **checked),
            GATE_C: box("sha256:img-hermes", ["HERMES_HOME=/opt/data"], [vol(HOME_V, "/opt/data")],
                        runtime=[10000, 10000], **checked),
            BROWSER_C: box("sha256:img-browser",
                           ["JOURNEY_BASE_URL=https://workspace.synthetic.test",
                            f"JOURNEY_PRODUCTION_ORIGIN={PROD_ORIGIN}", "PLAYWRIGHT_BROWSERS_PATH=/pw"], [],
                           **checked),
            PROD_C: box("sha256:img-prod", [f"RAPHAEL_ACCESS_ISSUER={PROD_ISSUER}"], []),
        }
        for name in DECOY_CONTAINERS:
            containers[name] = box("sha256:img-old", ["X=1"], [vol(DECOY_VOLUMES[0], "/state")])
        return {
            "hostname": STACK_HOST,
            "images": {k: {"env": v} for k, v in img.items()},
            "containers": containers,
            "volumes": STACK_VOLUMES + DECOY_VOLUMES,
            "curl": {"code": 403, "redirect": "", "rc": 0},
            "hermes": {"cron_jobs": {}, "send_rc": 1, "cron_rc": 0},
        }

    # --- who owns what in the fake volumes
    def set_identities(self, app: tuple[int, int], platform: tuple[int, int]) -> None:
        """The app user, the platform's runtime account, and the old volumes as the last boot left them."""
        def mutate(state: dict) -> None:
            state["containers"][APP_C]["user"] = list(app)
            for name in (DASH_C, GATE_C):
                state["containers"][name]["runtime"] = list(platform)
        if (self.root / "state.json").exists():
            self.update(mutate)
        owners = {}
        for volume, who in ((HOME_V, platform), (TOKENS_V, app)):
            for path in [self.vol(volume), *self.vol(volume).rglob("*")]:
                owners[str(path.relative_to(self.root / "volumes"))] = list(who)
        (self.root / "owners.json").write_text(json.dumps(owners))

    def owners(self) -> dict[str, list[int]]:
        return json.loads((self.root / "owners.json").read_text())

    def owner_of(self, path: Path) -> tuple[int, int]:
        """Modelled owner (uid, gid) of an entry of a volume; entries that are not listed belong to root."""
        found = self.owners().get(str(path.relative_to(self.root / "volumes")), [0, 0])
        return found[0], found[1]

    # --- reading and changing the fake host
    def state(self) -> dict:
        return json.loads((self.root / "state.json").read_text())

    def update(self, mutate) -> None:
        state = self.state()
        mutate(state)
        (self.root / "state.json").write_text(json.dumps(state))
        write_view(state, self.root)

    def vol(self, name: str) -> Path:
        return self.root / "volumes" / name

    def calls(self, tool: str | None = None) -> list[dict]:
        path = self.root / "calls.jsonl"
        if not path.exists():
            return []
        by_seq: dict[str, dict] = {}
        for line in path.read_text().splitlines():
            record = json.loads(line)
            if record.get("end"):
                by_seq[record["seq"]].update(rc=record["rc"], stdout=record["stdout"],
                                             stderr=record["stderr"])
            else:
                stdin_file = record.get("stdin_file")
                record["stdin"] = Path(stdin_file).read_bytes() if stdin_file else None
                by_seq[record["seq"]] = record
        return [c for c in by_seq.values() if tool is None or c["tool"] == tool]

    def clear_calls(self) -> None:
        for name in ("calls.jsonl", "chown.log"):
            (self.root / name).unlink(missing_ok=True)
        shutil.rmtree(self.root / "stdin", ignore_errors=True)

    def model_token_file(self, content: str = MODEL_TOKEN + "\n") -> Path:
        path = self.root / "model-token"
        path.write_text(content)
        return path

    def run(self, script: Path, *args: str, env: dict | None = None) -> subprocess.CompletedProcess:
        full = {
            "PATH": f"{self.root / 'bin'}:/usr/bin:/bin",
            "HOME": str(self.root / "home"),
            "FAKE_HOST_ROOT": str(self.root),
            "PYTHONDONTWRITEBYTECODE": "1",
            "JOURNEY_STACK_HOST": STACK_HOST,
            "JOURNEY_PRODUCTION_ORIGIN": PROD_ORIGIN,
        }
        for key, value in (env or {}).items():
            if value is None:
                full.pop(key, None)
            else:
                full[key] = value
        return subprocess.run([str(script), *args], env=full, cwd=self.root / "work",
                              capture_output=True, text=True, timeout=180)


@pytest.fixture
def host(tmp_path: Path) -> Host:
    return Host(tmp_path / "host")


ENV_SETTINGS = "env.settings"  # stack-home name for the profile settings; reset installs it as .env
SECRET_NAMES_IN_HOME = {"API_SERVER_KEY", "ANTHROPIC_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN"}


def snapshot(host: Host) -> dict[str, dict[str, str]]:
    return {v: sha_tree(host.vol(v)) for v in STACK_VOLUMES + DECOY_VOLUMES}


def assert_nothing_changed(host: Host, before: dict) -> None:
    """No stopping, emptying, writing, starting or restarting happened."""
    assert snapshot(host) == before
    for call in host.calls("docker"):
        sub, _ = docker_target(call)
        assert sub in ("inspect", "exec", "id-run"), f"changing docker call: {call['argv'][:3]}"
        if sub == "exec":
            assert "-i" not in call["argv"] and "--user" not in call["argv"], call["argv"][:6]
    assert not host.state().get("created_volumes")


def argv_text(host: Host) -> str:
    return "\n".join("\x00".join(call["argv"]) for call in host.calls())


def all_output(result: subprocess.CompletedProcess) -> str:
    return result.stdout + "\n" + result.stderr


def assert_secret_free(host: Host, result: subprocess.CompletedProcess, secrets: list[str]) -> None:
    assert secrets and all(len(s) >= 16 for s in secrets)
    container_output = "\n".join(c.get("stdout", "") + c.get("stderr", "") for c in host.calls("docker"))
    for secret in secrets:
        assert secret not in argv_text(host), "a secret reached a command line"
        assert secret not in all_output(result), "a secret was printed"
        assert secret not in container_output, "a container command printed a secret"


def home_files(host: Host) -> dict[str, Path]:
    home = host.vol(HOME_V)
    return {"default": home / ".env", "planner": home / "profiles/raphael-planner/.env"}


def generated_secrets(host: Host) -> list[str]:
    """Every access value the reset made, read back from the volumes."""
    found = []
    for path in host.vol(TOKENS_V).iterdir():
        found.append(path.read_text().strip())
    for path in home_files(host).values():
        for line in path.read_text().splitlines():
            if line.startswith(("API_SERVER_KEY=", "CLAUDE_CODE_OAUTH_TOKEN=")):
                found.append(line.partition("=")[2])
    return found


def run_reset(host: Host, token: bool = True, **env) -> subprocess.CompletedProcess:
    if token and "JOURNEY_MODEL_TOKEN_FILE" not in env:
        env["JOURNEY_MODEL_TOKEN_FILE"] = str(host.model_token_file())
    return host.run(RESET, env=env)


def reset_ok(host: Host, **env) -> subprocess.CompletedProcess:
    result = run_reset(host, **env)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def set_app_env(name: str, value: str | None):
    def mutate(state: dict) -> None:
        env = [e for e in state["containers"][APP_C]["env"] if not e.startswith(name + "=")]
        if value is not None:
            env.append(f"{name}={value}")
        state["containers"][APP_C]["env"] = env
    return mutate


def set_hermes_home(who: str, value: str):
    def mutate(state: dict) -> None:
        box = state["containers"][who]
        box["env"] = [f"HERMES_HOME={value}" if e.startswith("HERMES_HOME=") else e for e in box["env"]]
    return mutate


def mount_source(call: dict) -> str:
    spec = call["argv"][call["argv"].index("--mount") + 1]
    return dict(part.split("=", 1) for part in spec.split(","))["source"]


def parse_env_file(path: Path) -> tuple[list[str], dict[str, str]]:
    lines = [ln for ln in path.read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    return [ln.partition("=")[0] for ln in lines], dict(ln.partition("=")[::2] for ln in lines)


class SharedReset(NamedTuple):
    host: Host
    result: subprocess.CompletedProcess
    untouched: dict
    repo_before: dict


@pytest.fixture(scope="module")
def shared_reset(tmp_path_factory) -> SharedReset:
    """One default reset, run once; the tests that only read its outcome share it (and never change it)."""
    first = Host(tmp_path_factory.mktemp("after-reset") / "host")
    untouched = {v: sha_tree(first.vol(v)) for v in [PROXY_V] + DECOY_VOLUMES}
    repo_before = sha_tree(STACK_HOME)
    hermes_gives(real_platform=True)(first)  # the anchor tests read what the platform's code wrote in this reset
    result = reset_ok(first)
    first.update(lambda state: state["hermes"].pop("real_platform"))  # the copies of this host reset without it
    return SharedReset(first, result, untouched, repo_before)


# --- the fake itself: it enforces the exec user, the boot ownership and the owner-only rules


def docker_by_hand(host: Host, *args: str, stdin: str | None = None) -> subprocess.CompletedProcess:
    env = {"PATH": f"{host.root / 'bin'}:/usr/bin:/bin", "FAKE_HOST_ROOT": str(host.root),
           "HOME": str(host.root / "home"), "PYTHONDONTWRITEBYTECODE": "1"}
    return subprocess.run([str(host.root / "bin/docker"), *args], env=env, input=stdin, capture_output=True, text=True)


def own_volume(host: Host, volume: str, who: tuple[int, int], mode: int | None = None) -> None:
    table = host.owners()
    table[volume] = list(who)
    (host.root / "owners.json").write_text(json.dumps(table))
    if mode is not None:
        host.vol(volume).chmod(mode)


def writer_calls(host: Host) -> list[dict]:
    return [c for c in host.calls("docker") if docker_target(c)[0] == "exec" and "--user" in c["argv"]]


@pytest.fixture(scope="module")
def reset_tree(shared_reset: SharedReset) -> Host:
    """The reset stack that every check test below copies; it is never changed."""
    return shared_reset.host


@pytest.fixture
def stack(reset_tree: Host, tmp_path: Path) -> Host:
    return Host.copy_of(reset_tree, tmp_path / "host")


class CleanCheck(NamedTuple):
    host: Host
    before: dict
    result: subprocess.CompletedProcess


@pytest.fixture(scope="module")
def clean_check(reset_tree: Host, tmp_path_factory) -> CleanCheck:
    """One check run on a clean copy of the reset stack, shared by the tests that only read it."""
    clean = Host.copy_of(reset_tree, tmp_path_factory.mktemp("clean-check") / "host")
    before = snapshot(clean)
    return CleanCheck(clean, before, run_check(clean))


def run_check(host: Host, mode: str = "server2", **env) -> subprocess.CompletedProcess:
    return host.run(CHECK, mode, env=env)


def verdicts(result: subprocess.CompletedProcess) -> dict[str, str]:
    """Check name -> PASS or FAIL, from stdout lines that must have the agreed shape."""
    found: dict[str, str] = {}
    for line in result.stdout.splitlines():
        passed = re.fullmatch(r"PASS ([a-z0-9-]+)", line)
        failed = re.fullmatch(r"FAIL ([a-z0-9-]+): \S.*", line)
        assert passed or failed, f"unexpected output line: {line!r}"
        found[(passed or failed).group(1)] = "PASS" if passed else "FAIL"
    return found


def failed_checks(result: subprocess.CompletedProcess) -> set[str]:
    return {name for name, verdict in verdicts(result).items() if verdict == "FAIL"}


def profile_folder(host: Host, profile: str) -> Path:
    home = host.vol(HOME_V)
    return home if profile == "default" else home / "profiles" / f"raphael-{profile}"


def edit_config(host: Host, profile: str, change) -> None:
    path = profile_folder(host, profile) / "config.yaml"
    data = yaml.safe_load(path.read_text()) or {}
    change(data)
    path.write_text(yaml.safe_dump(data))


def dispatcher_on(profile: str):
    return lambda host: edit_config(
        host, profile, lambda d: d.setdefault("kanban", {}).update(dispatch_in_gateway=True))


def dispatcher_unset(profile: str):
    return lambda host: edit_config(host, profile, lambda d: d.pop("kanban", None))


def cron_job(profile: str):
    return lambda host: host.update(lambda s: s["hermes"]["cron_jobs"].update({profile: ["job-1"]}))


def platform_enabled(profile: str):
    return lambda host: edit_config(
        host, profile, lambda d: d.setdefault("platforms", {}).update(discord={"enabled": True}))


def hermes_gives(**settings):
    return lambda host: host.update(lambda s: s["hermes"].update(settings))


NO_STACK_ENV = {"JOURNEY_STACK_HOST": None, "JOURNEY_PRODUCTION_ORIGIN": None}
