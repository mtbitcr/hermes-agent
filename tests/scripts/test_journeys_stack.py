"""Tests for the journeys stack reset and isolation scripts (card P-2).

The scripts run against a fake docker, hostname, curl and sleep that sit first on PATH in a temp
folder. The "host" is a folder tree: containers and volumes are entries in a state file, volumes
are plain folders, and the shell snippets the scripts send into containers really run locally on
those folders. No docker, no real Server 2 and no outside network is involved. The fake curl never
reads a default curl configuration, so two tests of the production probe put the real curl in its
place (behind a wrapper that records its argument list) against a throw-away HTTPS endpoint on
127.0.0.1, under a temporary HOME whose .curlrc asks for a POST.
"""

from __future__ import annotations

import base64
import datetime
import hashlib
import http.server
import io
import ipaddress
import json
import os
import re
import shutil
import ssl
import stat
import subprocess
import sys
import tarfile
import tempfile
import threading
from pathlib import Path
from typing import NamedTuple

import pytest
import yaml

from plugins.dashboard_auth.raphael_workspace import model_policy

pytestmark = pytest.mark.linux_only

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
                        "F_HERMES_HOME", "F_HEALTH_KIND", "F_APP_SETTINGS", "TOKEN_VARS"))
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


# ------------------------------------------------------------------ files in the repository


@pytest.mark.parametrize("script", [RESET, CHECK], ids=lambda p: p.name)
def test_scripts_are_executable_bash(script: Path):
    assert script.is_file(), f"{script.name} is missing"
    assert os.access(script, os.X_OK)
    assert script.read_text().splitlines()[0] == "#!/usr/bin/env bash"
    assert subprocess.run(["bash", "-n", str(script)], capture_output=True).returncode == 0


def test_template_holds_names_only():
    sections = parse_template()
    assert sections
    for section, names in sections.items():
        assert len(names) == len(set(names)), f"duplicate name in [{section}]"
        for name in names:
            assert re.fullmatch(r"[A-Za-z0-9_.-]+", name), f"[{section}] holds more than a name: {name!r}"


def test_template_lists_the_stack_names():
    sections = parse_template()
    assert sorted(sections["containers"]) == sorted(STACK_CONTAINERS)
    assert sorted(sections["volumes"]) == sorted(STACK_VOLUMES)
    assert set(sections["home-env"]) == SECRET_NAMES_IN_HOME
    assert set(sections[f"env {APP_C}"]) == set(APP_ENV_NAMES)
    assert sections[f"env {PROXY_C}"] == ["SYNTHETIC_STATE"]
    assert sections[f"env {DASH_C}"] == ["HERMES_HOME"]
    assert sections[f"env {GATE_C}"] == ["HERMES_HOME"]
    assert set(sections[f"env {BROWSER_C}"]) == {
        "JOURNEY_BASE_URL", "JOURNEY_PRODUCTION_ORIGIN", "PLAYWRIGHT_BROWSERS_PATH"}
    assert set(sections["script-input"]) == {
        "JOURNEY_STACK_HOST", "JOURNEY_PRODUCTION_ORIGIN", "JOURNEY_MODEL_TOKEN_FILE"}
    assert sections["production-container"] == [PROD_C]
    assert sections["production-read-variable"] == ["RAPHAEL_ACCESS_ISSUER"]
    expected = {"containers", "volumes", "home-env", "script-input", "production-container",
                "production-read-variable"} | {f"env {c}" for c in (APP_C, PROXY_C, DASH_C, GATE_C, BROWSER_C)}
    assert set(sections) == expected


def test_template_leaves_out_github_and_page_check_names():
    every_name = {name for names in parse_template().values() for name in names}
    assert not every_name & {"RAPHAEL_GITHUB_APP_ID", "RAPHAEL_GITHUB_INSTALLATION_ID",
                             "RAPHAEL_GITHUB_PRIVATE_KEY_FILE", "RAPHAEL_PAGE_CHECK_COMMON_NAME"}
    assert not every_name & (set(DECOY_CONTAINERS) | set(DECOY_VOLUMES))


def test_template_says_in_one_comment_line_that_every_container_needs_a_health_check():
    lines = [line for line in TEMPLATE.read_text().splitlines() if "health check" in line.lower()]
    assert len(lines) == 1 and lines[0].startswith("#"), lines
    assert "docker" in lines[0].lower() and "every container" in lines[0].lower()


def test_stack_home_sets_what_the_facts_say():
    default = yaml.safe_load((STACK_HOME / "config.yaml").read_text())
    profiles = sorted(p for p in (STACK_HOME / "profiles").iterdir() if p.is_dir())
    assert [p.name for p in profiles] == ["raphael-designer", "raphael-planner"]
    designer, planner = (yaml.safe_load((p / "config.yaml").read_text()) for p in profiles)
    multiplex = default.get("multiplex_profiles", (default.get("gateway") or {}).get("multiplex_profiles"))
    assert multiplex is True
    for config in (default, planner, designer):
        assert config["kanban"]["dispatch_in_gateway"] is False
        platforms = config.get("platforms") or {}
        assert set(platforms) <= {"api_server"}, "no messaging platform may be configured"
        assert "cron" not in config


def primary_route(profile: str):
    """The admitted base route the policy recommends for a profile, read from the policy when the test runs."""
    found = [a for (name, _provider), a in model_policy._ASSIGNMENTS.items() if name == profile and a.recommended]
    assert len(found) == 1, f"the policy has no single primary route for {profile}"
    return found[0]


def test_stack_home_enables_the_owner_workspace_and_carries_the_admitted_route(tmp_path: Path, monkeypatch):
    configs = {
        "default": STACK_HOME / "config.yaml",
        "raphael-planner": STACK_HOME / "profiles/raphael-planner/config.yaml",
    }
    def dig(node, *keys):
        for key in keys:
            node = node.get(key) if isinstance(node, dict) else None
        return node

    for profile, path in configs.items():
        config = yaml.safe_load(path.read_text())
        route = primary_route(profile)
        assert dig(config, "gateway", "api_server", "owner_workspace", "enabled") is True, profile
        assert dig(config, "model", "provider") == route.provider, profile
        assert dig(config, "model", "default") == route.model, profile
        assert dig(config, "agent", "reasoning_effort") == route.reasoning_effort, profile
        assert config.get("fallback_providers") == [], profile

    # What the plan commit reads: the policy's own reader, on a copy of the home, gives that route and no fallback.
    home = tmp_path / "home"
    shutil.copytree(STACK_HOME, home)
    monkeypatch.setenv("HERMES_HOME", str(home))
    for profile in configs:
        route = primary_route(profile)
        configured = model_policy._configured_route(profile)
        assert configured == (route.provider, route.model, route.reasoning_effort, True), profile
        provider, model, effort, fallbacks_disabled = configured
        admitted = model_policy.validate_assignment(profile, provider, model, effort, disable_fallbacks=fallbacks_disabled)
        assert admitted == route, profile


def test_stack_home_holds_no_dotenv_and_no_values():
    files = [p for p in STACK_HOME.rglob("*") if p.is_file()]
    assert files
    assert not [p for p in files if p.name == ".env"], ".gitignore ignores any file named .env"
    assert sorted(p.relative_to(STACK_HOME).as_posix() for p in files if p.name == ENV_SETTINGS) == [
        ENV_SETTINGS, f"profiles/raphael-planner/{ENV_SETTINGS}"]
    allowed = set(parse_template()["home-env"])
    for path in files:
        for line in path.read_text().splitlines():
            assigned = re.match(r"\s*([A-Z][A-Z0-9_]*)\s*=(.*)$", line)
            if assigned:
                assert assigned.group(1) in allowed, f"{path.name}: name not on the template"
                assert not assigned.group(2).strip(), f"{path.name}: a value is set"


def test_owned_paths_are_not_git_ignored(tmp_path: Path):
    assert RESET.is_file() and CHECK.is_file() and TEMPLATE.is_file(), "an owned file is missing"
    assert [p for p in STACK_HOME.rglob("*") if p.is_file()], "stack-home holds no files"
    repo = tmp_path / "ignore-check"
    repo.mkdir()
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(tmp_path),
           "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_NOSYSTEM": "1"}
    subprocess.run(["git", "init", "-q"], cwd=repo, env=env, check=True, capture_output=True)
    shutil.copy(REPO / ".gitignore", repo / ".gitignore")
    owned = [RESET, CHECK, TEMPLATE, Path(__file__).resolve()]
    owned += [p for p in STACK_HOME.rglob("*") if p.is_file()]
    for path in owned:
        rel = path.relative_to(REPO)
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text("x\n")
        result = subprocess.run(["git", "check-ignore", "-q", str(rel)], cwd=repo, env=env,
                                capture_output=True)
        assert result.returncode == 1, f"{rel} is ignored by .gitignore (or git failed)"
    # The guard itself works: a file named .env is ignored.
    (repo / "scripts/journeys/stack-home/.env").parent.mkdir(parents=True, exist_ok=True)
    (repo / "scripts/journeys/stack-home/.env").write_text("x\n")
    assert subprocess.run(["git", "check-ignore", "-q", "scripts/journeys/stack-home/.env"],
                          cwd=repo, env=env, capture_output=True).returncode == 0


# ------------------------------------------------------------------ helpers for the script tests


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


# ------------------------------------------------------------------ reset_stack.sh: refusals


def run_reset(host: Host, token: bool = True, **env) -> subprocess.CompletedProcess:
    if token and "JOURNEY_MODEL_TOKEN_FILE" not in env:
        env["JOURNEY_MODEL_TOKEN_FILE"] = str(host.model_token_file())
    return host.run(RESET, env=env)


def reset_ok(host: Host, **env) -> subprocess.CompletedProcess:
    result = run_reset(host, **env)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


@pytest.mark.parametrize("stack_host", ["some-other-host.invalid", "", None])
def test_reset_refuses_the_wrong_host(host: Host, stack_host):
    before = snapshot(host)
    result = run_reset(host, JOURNEY_STACK_HOST=stack_host)
    assert result.returncode != 0
    assert_nothing_changed(host, before)
    assert STACK_HOST not in all_output(result)


@pytest.mark.parametrize("missing", STACK_CONTAINERS)
def test_reset_refuses_when_a_container_is_missing(host: Host, missing: str):
    host.update(lambda state: state["containers"].pop(missing))
    before = snapshot(host)
    result = run_reset(host)
    assert result.returncode != 0
    assert missing in all_output(result)
    assert_nothing_changed(host, before)


@pytest.mark.parametrize("stopped", [APP_C, DASH_C])
def test_reset_refuses_when_the_app_or_dashboard_is_not_running(host: Host, stopped: str):
    host.update(lambda state: state["containers"][stopped].update(running=False))
    before = snapshot(host)
    result = run_reset(host)
    if stopped == APP_C:  # a stopped owner app is no longer refused: the reset restarts it
        assert result.returncode == 0, result.stdout + result.stderr
        assert host.state()["containers"][APP_C]["running"]
        return
    assert result.returncode != 0
    assert_nothing_changed(host, before)


def set_app_env(name: str, value: str | None):
    def mutate(state: dict) -> None:
        env = [e for e in state["containers"][APP_C]["env"] if not e.startswith(name + "=")]
        if value is not None:
            env.append(f"{name}={value}")
        state["containers"][APP_C]["env"] = env
    return mutate


@pytest.mark.parametrize("name", list(SURFACE_ENV.values()) + list(KEY_ENV.values()))
def test_reset_refuses_a_token_path_outside_the_app_mount(host: Host, name: str):
    host.update(set_app_env(name, "/elsewhere/token"))
    before = snapshot(host)
    result = run_reset(host)
    assert result.returncode != 0
    assert_nothing_changed(host, before)


@pytest.mark.parametrize("value", ["/run/tokens/../models", "models", "/run/tokensX/models", "", None])
def test_reset_refuses_odd_token_paths(host: Host, value):
    host.update(set_app_env(SURFACE_ENV["models"], value))
    before = snapshot(host)
    result = run_reset(host)
    assert result.returncode != 0
    assert_nothing_changed(host, before)


@pytest.mark.parametrize("who", [APP_C, DASH_C])
def test_reset_refuses_when_a_container_lacks_the_token_volume(host: Host, who: str):
    def mutate(state: dict) -> None:
        box = state["containers"][who]
        box["mounts"] = [m for m in box["mounts"] if m["name"] != TOKENS_V]
    host.update(mutate)
    before = snapshot(host)
    result = run_reset(host)
    assert result.returncode != 0
    assert_nothing_changed(host, before)


def set_hermes_home(who: str, value: str):
    def mutate(state: dict) -> None:
        box = state["containers"][who]
        box["env"] = [f"HERMES_HOME={value}" if e.startswith("HERMES_HOME=") else e for e in box["env"]]
    return mutate


@pytest.mark.parametrize("value", ["/somewhere/else", "/opt/data/", "/opt/dat"])
@pytest.mark.parametrize("who", [DASH_C, GATE_C])
def test_reset_refuses_when_hermes_home_is_not_the_home_mount(host: Host, who: str, value: str):
    host.update(set_hermes_home(who, value))
    before = snapshot(host)
    result = run_reset(host)
    assert result.returncode != 0
    assert who in result.stderr and "nothing was changed" in result.stderr
    assert_nothing_changed(host, before)


def test_reset_refuses_a_gateway_home_that_is_another_directory_with_a_stale_state(host: Host):
    other = host.root / "gateway-own-filesystem" / "data"
    shutil.copytree(STACK_HOME, other)
    (other / ".env").write_text("API_SERVER_KEY=stale-gateway-key-0123456789abcdef\n")
    (other / "stale-state.db").write_text("sentinel rows that must stay\n")
    host.update(set_hermes_home(GATE_C, str(other)))
    before, stale = snapshot(host), sha_tree(other)
    result = run_reset(host)
    assert result.returncode != 0
    assert GATE_C in result.stderr and "nothing was changed" in result.stderr
    assert_nothing_changed(host, before)
    assert sha_tree(other) == stale and (other / "stale-state.db").read_text() == "sentinel rows that must stay\n"


def test_reset_reads_hermes_home_of_both_containers_with_one_inspect_format(shared_reset: SharedReset):
    calls = shared_reset.host.calls("docker")
    read = [c["argv"][-1] for c in calls if c["argv"][0] == "inspect" and F_HERMES_HOME in c["argv"]]
    assert sorted(read) == sorted([DASH_C, GATE_C])
    assert not [c for c in calls if c["argv"][0] == "exec" and "HERMES_HOME" in c["argv"]]


def test_reset_does_not_refuse_a_stopped_gateway_whose_hermes_home_is_its_mount(host: Host):
    host.update(lambda state: state["containers"][GATE_C].update(running=False))
    result = run_reset(host)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "HERMES_HOME" not in all_output(result)


def unreadable(who: str, command: str, text: str):
    def mutate(state: dict) -> None:
        state["containers"][who].setdefault("answers", {})[command] = text
    return mutate


@pytest.mark.parametrize("who, command, text", [
    (APP_C, "id -u", "node"), (APP_C, "id -u", "0"), (APP_C, "id -u", ""),
    (APP_C, "id -g", "node"), (APP_C, "id -g", "0"), (APP_C, "id -g", ""),
    (DASH_C, "stat -c %u:%g /opt/data", "root:root"), (DASH_C, "stat -c %u:%g /opt/data", "0:0"),
    (DASH_C, "stat -c %u:%g /opt/data", "10000:0"), (DASH_C, "stat -c %u:%g /opt/data", "0:10000"),
    (DASH_C, "stat -c %u:%g /opt/data", ""), (DASH_C, "stat -c %u:%g /opt/data", "10000"),
], ids=lambda value: str(value).replace(" ", "_") or "empty")
def test_reset_refuses_an_identity_it_cannot_read_as_two_positive_numbers(host: Host, who, command, text):
    host.update(unreadable(who, command, text))
    before = snapshot(host)
    result = run_reset(host)
    assert result.returncode != 0
    assert who in result.stderr and "nothing was changed" in result.stderr
    assert_nothing_changed(host, before)


def test_reset_refuses_when_the_gateway_lacks_the_home_volume(host: Host):
    host.update(lambda state: state["containers"][GATE_C].update(mounts=[]))
    before = snapshot(host)
    result = run_reset(host)
    assert result.returncode != 0
    assert_nothing_changed(host, before)


# ------------------------------------------------------------------ reset_stack.sh: a full run


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
    return SharedReset(first, reset_ok(first), untouched, repo_before)


def test_reset_makes_a_fresh_home_and_token_folder(shared_reset: SharedReset):
    host, result, untouched = shared_reset.host, shared_reset.result, shared_reset.untouched
    home, tokens = host.vol(HOME_V), host.vol(TOKENS_V)

    assert not (home / "stale").exists() and not (home / "keep.txt").exists()
    assert not list(home.rglob(ENV_SETTINGS))
    for path in STACK_HOME.rglob("*"):
        if path.is_file() and path.name != ENV_SETTINGS:
            assert (home / path.relative_to(STACK_HOME)).read_bytes() == path.read_bytes()
    assert stat.S_IMODE(home.stat().st_mode) & 0o077 == 0

    allowed = set(parse_template()["home-env"])
    keys = {}
    for profile, path in home_files(host).items():
        assert path.is_file() and not path.is_symlink()
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        names, values = parse_env_file(path)
        assert set(names) <= allowed and names.count("API_SERVER_KEY") == 1
        assert len(values["API_SERVER_KEY"]) >= 16
        assert values["CLAUDE_CODE_OAUTH_TOKEN"] == MODEL_TOKEN
        keys[profile] = values["API_SERVER_KEY"]
    assert keys["default"] != keys["planner"]

    assert stat.S_IMODE(tokens.stat().st_mode) == 0o700
    assert {p.name for p in tokens.iterdir()} == set(SURFACE_ENV) | {"executor", "planner"}
    for path in tokens.iterdir():
        info = path.lstat()
        assert stat.S_ISREG(info.st_mode) and stat.S_IMODE(info.st_mode) == 0o600
        assert 0 < info.st_size <= 16 * 1024
    for surface in SURFACE_ENV:
        assert (tokens / surface).read_text().startswith(f"tok-{surface}-")
    assert (tokens / "executor").read_text().strip() == keys["default"]
    assert (tokens / "planner").read_text().strip() == keys["planner"]

    assert {v: sha_tree(host.vol(v)) for v in untouched} == untouched
    assert not host.state().get("created_volumes")
    assert_secret_free(host, result, generated_secrets(host))


def test_reset_can_run_twice_and_makes_new_values(shared_reset: SharedReset, tmp_path: Path):
    first = generated_secrets(shared_reset.host)
    assert_secret_free(shared_reset.host, shared_reset.result, first)
    host = Host.copy_of(shared_reset.host, tmp_path / "host")
    result = reset_ok(host)
    second = generated_secrets(host)
    assert len(second) == len(first)
    assert not (set(first) - {MODEL_TOKEN}) & set(second)
    for path in home_files(host).values():
        assert parse_env_file(path)[0].count("API_SERVER_KEY") == 1
    assert_secret_free(host, result, first + second)


def test_reset_order_of_stops_writes_and_starts(shared_reset: SharedReset):
    host = shared_reset.host
    steps = []
    for call in host.calls("docker"):
        sub, target = docker_target(call)
        if sub in ("stop", "start", "restart"):
            steps.append((sub, target))
        elif sub == "run":
            steps.append(("helper", mount_source(call)))
        elif sub == "exec" and "--user" in call["argv"]:
            steps.append(("write", target))
    assert set(steps[:2]) == {("stop", GATE_C), ("stop", DASH_C)}
    assert set(steps[2:4]) == {("helper", HOME_V), ("helper", TOKENS_V)}
    assert steps[4] == ("start", DASH_C)
    # the anchor Project and its board, then 5 tokens, 2 API server keys, 2 model tokens
    assert steps[5:-6] == [("write", DASH_C)] * (ANCHOR_WRITES + 9)
    assert steps[-6] == ("helper", TOKENS_V)  # the token files go to the owner app before it restarts
    assert steps[-5:-2] == [("restart", APP_C), ("restart", DASH_C), ("start", GATE_C)]
    assert set(steps[-2:]) == {("start", PROXY_C), ("start", BROWSER_C)}


def test_reset_helpers_have_the_agreed_shape(shared_reset: SharedReset):
    host = shared_reset.host
    image = host.state()["containers"][DASH_C]["image"]
    runs = [c for c in host.calls("docker") if c["argv"][0] == "run"]
    assert sorted(mount_source(c) for c in runs) == sorted([HOME_V, TOKENS_V, TOKENS_V])
    for call in runs:
        argv, source = call["argv"], mount_source(call)
        flags = ["--rm", "-i"] if source == HOME_V else ["--rm"]
        assert argv[:-1] == ["run", *flags, "--network", "none", "--user", "0", "--entrypoint", "sh",
                             "--mount", f"type=volume,source={source},target=/volume", image, "-c"]
        script = argv[-1]
        if source == HOME_V:
            assert "find /volume -mindepth 1 -delete" in script
            assert "chown -R 10000:10000 /volume" in script and "chmod go-rwx /volume" in script
            with tarfile.open(fileobj=io.BytesIO(call["stdin"])) as archive:
                members = {m.name.removeprefix("./") for m in archive.getmembers() if m.isfile()}
            assert members == {p.relative_to(STACK_HOME).as_posix() for p in STACK_HOME.rglob("*")
                               if p.is_file()}
        elif "-delete" in script:
            assert "chown 10000:10000 /volume" in script and "chmod 700 /volume" in script
            assert not call["stdin"]
        else:
            assert script.splitlines()[-1] == "chown -R 1000:1000 /volume"
            assert not call["stdin"]


def test_reset_writers_run_as_the_platform_identity_with_values_only_on_stdin(shared_reset: SharedReset):
    host = shared_reset.host
    writers = [c for c in host.calls("docker") if docker_target(c)[0] == "exec" and "--user" in c["argv"]]
    assert len(writers) == ANCHOR_WRITES + 9
    for call in writers:
        argv = call["argv"]
        assert argv[argv.index("--user") + 1] == "10000:10000"
        assert docker_target(call)[1] == DASH_C
    issues = [c for c in writers if "kanban-workspace-token" in c["argv"]]
    assert sorted(c["argv"][c["argv"].index("--surface") + 1] for c in issues) == sorted(SURFACE_ENV)
    for call in issues:
        argv = call["argv"]
        assert not [a for a in argv if a.startswith("--ttl")]
        assert argv[argv.index("--out") + 1] == "/tokens/" + argv[argv.index("--surface") + 1]
    carrying = [c for c in writers if c["stdin"] and MODEL_TOKEN.encode() in c["stdin"]]
    assert len(carrying) == 2 and all("-i" in c["argv"] for c in carrying)


# --- the Project and board that the owner app reads

# The platform's own registry and kanban code, run on the home that the reset left: what is there, as the platform reads it.
ANCHOR_READ = (
    "import json, sys\n"
    "from hermes_cli import kanban_db, projects_db\n"
    "project_slug, board_slug = sys.argv[1:3]\n"
    "with projects_db.connect_closing() as conn:\n"
    "    project = projects_db.get_project(conn, project_slug)\n"
    "meta = kanban_db.read_board_metadata(board_slug)\n"
    "print(json.dumps({'project': project and {'id': project.id, 'slug': project.slug, 'board': project.board_slug},\n"
    "                  'board_exists': kanban_db.board_exists(board_slug),\n"
    "                  'meta': {'slug': meta.get('slug'), 'project_id': meta.get('project_id')}}))\n")


def bound_hermes_source() -> tuple[str, str]:
    """The Project slug and the board slug that the owner app's settings bind: what the app reads, and through which board."""
    lines = (JOURNEYS / "stack-app.settings").read_text().splitlines()
    settings = dict(line.partition("=")[::2] for line in lines if line.strip() and not line.startswith("#"))
    (binding,) = json.loads(settings["RAPHAEL_PROJECT_BINDINGS_JSON"])
    hermes = binding["sources"]["hermes"]
    return hermes["projectSlug"], hermes["boardSlug"]


def read_anchor(host: Host, scratch: Path) -> dict:
    """The bound Project and board as the platform reads them from a copy of the home that the reset left."""
    project_slug, board_slug = bound_hermes_source()
    home = scratch / "home-copy"
    shutil.copytree(host.vol(HOME_V), home, symlinks=True)
    REAL_PYC.mkdir(parents=True, exist_ok=True)
    env = {"PATH": os.environ["PATH"], "HOME": str(scratch), "HERMES_HOME": str(home), "PYTHONPATH": str(REPO),
           "PYTHONPYCACHEPREFIX": str(REAL_PYC), "AWS_EC2_METADATA_DISABLED": "true"}
    done = subprocess.run([sys.executable, "-c", ANCHOR_READ, project_slug, board_slug], env=env, cwd=scratch,
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stdout + done.stderr
    return json.loads(done.stdout.splitlines()[-1])


def test_reset_makes_the_project_the_app_binds_and_a_board_that_names_that_projects_id(
        shared_reset: SharedReset, tmp_path: Path):
    project_slug, board_slug = bound_hermes_source()
    seen = read_anchor(shared_reset.host, tmp_path)
    assert seen["project"] is not None, f"the reset made no Project with the slug {project_slug}"
    assert seen["project"]["slug"] == project_slug and seen["project"]["board"] == board_slug
    assert seen["board_exists"], f"the reset made no board {board_slug}"
    assert seen["meta"]["slug"] == board_slug
    assert seen["meta"]["project_id"] == seen["project"]["id"]


def test_reset_makes_the_project_and_board_before_it_issues_any_token(shared_reset: SharedReset):
    writers = writer_calls(shared_reset.host)
    issues = [i for i, call in enumerate(writers) if "kanban-workspace-token" in call["argv"]]
    assert len(issues) == len(SURFACE_ENV)
    assert min(issues) >= ANCHOR_WRITES


def board_root_is_a_file(host: Host) -> None:
    """The platform's kanban folder is pinned to a file, so that no board can be written (the Project registry is not affected)."""
    blocker = host.root / "not-a-folder"
    blocker.write_text("a file where the platform expects the kanban folder\n")
    host.update(lambda state: state["containers"][DASH_C]["env"].append(f"HERMES_KANBAN_HOME={blocker}"))


def project_command_gives(mode: str):
    return lambda host: host.update(lambda state: state["hermes"].update(project_fail=mode))


# One operation broken per case: how to break it, and whether the Project was made before the reset stopped.
ANCHOR_BREAKS = {
    "project-command-refused": (project_command_gives("refused"), False),
    "project-command-makes-nothing": (project_command_gives("silent"), False),
    "board-cannot-be-written": (board_root_is_a_file, True),
}


@pytest.mark.parametrize("name", ANCHOR_BREAKS)
def test_reset_stops_before_any_token_when_the_project_or_its_board_cannot_be_made(host: Host, tmp_path: Path, name):
    break_it, project_made = ANCHOR_BREAKS[name]
    break_it(host)
    result = run_reset(host)
    assert result.returncode != 0, result.stdout + result.stderr
    docker = host.calls("docker")
    assert not [c for c in docker if "kanban-workspace-token" in c["argv"]]
    done = [docker_target(c) for c in docker]
    assert not [step for step in done if step[0] == "restart"] and ("start", GATE_C) not in done
    seen = read_anchor(host, tmp_path)
    assert (seen["project"] is not None) == project_made
    assert not seen["board_exists"] and seen["meta"]["project_id"] is None


# --- the platform's runtime identity and the owner app's identity are two different things

IDENTITY_CASES = [
    pytest.param("shared_reset", (1000, 1000), (10000, 10000), id="platform-10000-app-1000"),
    pytest.param("reset_remapped", (1000, 1000), (4242, 4242), id="platform-4242-app-1000"),
    pytest.param("reset_equal", (1000, 1000), (1000, 1000), id="platform-1000-app-1000"),
]


def reset_with_identities(tmp_path_factory, app: tuple[int, int], platform: tuple[int, int]) -> SharedReset:
    first = Host(tmp_path_factory.mktemp("identities") / "host")
    first.set_identities(app, platform)
    return SharedReset(first, run_reset(first), {}, {})


@pytest.fixture(scope="module")
def reset_remapped(tmp_path_factory) -> SharedReset:
    return reset_with_identities(tmp_path_factory, (1000, 1000), (4242, 4242))


@pytest.fixture(scope="module")
def reset_equal(tmp_path_factory) -> SharedReset:
    return reset_with_identities(tmp_path_factory, (1000, 1000), (1000, 1000))


def finished(request, fixture: str) -> SharedReset:
    shared = request.getfixturevalue(fixture)
    assert shared.result.returncode == 0, shared.result.stdout + shared.result.stderr
    return shared


@pytest.mark.parametrize("fixture, app, platform", IDENTITY_CASES)
def test_reset_succeeds_whether_the_platform_and_app_identities_differ_or_match(request, fixture, app, platform):
    assert "the stack is reset" in finished(request, fixture).result.stdout


@pytest.mark.parametrize("fixture, app, platform", IDENTITY_CASES)
def test_reset_gives_the_token_files_to_the_app_identity_in_an_owner_only_folder(request, fixture, app, platform):
    host = finished(request, fixture).host
    tokens = host.vol(TOKENS_V)
    assert host.owner_of(tokens) == app and stat.S_IMODE(tokens.stat().st_mode) == 0o700
    names = set(SURFACE_ENV) | {"executor", "planner"}
    assert {p.name for p in tokens.iterdir()} == names
    for name in names:
        info = (tokens / name).lstat()
        assert host.owner_of(tokens / name) == app
        assert stat.S_ISREG(info.st_mode) and stat.S_IMODE(info.st_mode) == 0o600 and 0 < info.st_size <= 16 * 1024


@pytest.mark.parametrize("fixture, app, platform", IDENTITY_CASES)
def test_reset_leaves_the_home_and_its_files_with_the_platform_identity(request, fixture, app, platform):
    host = finished(request, fixture).host
    home = host.vol(HOME_V)
    for path in [home, *home.rglob("*")]:
        assert host.owner_of(path) == platform, path
    assert stat.S_IMODE(home.stat().st_mode) & 0o077 == 0
    for path in home_files(host).values():
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    for path in (home / "config.yaml", home / "profiles/raphael-planner/config.yaml"):
        assert path.is_file()


@pytest.mark.parametrize("fixture, app, platform", IDENTITY_CASES)
def test_reset_runs_every_platform_command_as_the_platform_identity_and_nothing_as_the_app(
        request, fixture, app, platform):
    host = finished(request, fixture).host
    calls, spec = host.calls("docker"), "%d:%d" % platform
    writers = writer_calls(host)
    assert len(writers) == ANCHOR_WRITES + 9
    for call in writers:
        assert call["argv"][call["argv"].index("--user") + 1] == spec
    for call in calls:
        argv = call["argv"]
        if argv[0] == "run":
            assert argv[argv.index("--user") + 1] == "0"
        elif argv[0] == "exec" and "--user" not in argv:
            assert argv[2:4] in (["id", "-u"], ["id", "-g"], ["printenv", argv[3]]) or argv[2:4] == ["stat", "-c"]
    app_spec = "%d:%d" % app
    named = [c for c in calls if re.search(rf"(?<![0-9]){app_spec}(?![0-9])", "\x00".join(c["argv"]))]
    if app == platform:
        return
    assert len(named) == 1 and named[0]["argv"][0] == "run" and mount_source(named[0]) == TOKENS_V
    assert named[0]["argv"][-1].splitlines()[-1] == f"chown -R {app_spec} /volume"


def test_reset_reads_both_identities_as_numbers_before_it_changes_anything(shared_reset: SharedReset):
    argvs = [c["argv"] for c in shared_reset.host.calls("docker")]
    first_change = next(i for i, argv in enumerate(argvs) if argv[0] in ("stop", "start", "restart", "run"))
    seen = argvs[:first_change]
    assert ["exec", APP_C, "id", "-u"] in seen and ["exec", APP_C, "id", "-g"] in seen
    assert ["exec", DASH_C, "stat", "-c", "%u:%g", "/opt/data"] in seen


def test_reset_reads_a_running_owner_app_only_by_docker_exec_before_it_changes_anything(shared_reset: SharedReset):
    argvs = [c["argv"] for c in shared_reset.host.calls("docker")]
    first_change = next(i for i, argv in enumerate(argvs) if argv[0] in ("stop", "start", "restart", "run"))
    into_app = [argv for argv in argvs if argv[:2] == ["exec", APP_C]]
    assert into_app == [argv for argv in argvs[:first_change] if argv[:2] == ["exec", APP_C]]
    assert into_app == [["exec", APP_C, "id", "-u"], ["exec", APP_C, "id", "-g"],
                        *(["exec", APP_C, "printenv", name] for name in TOKEN_VARS)]
    assert not [argv for argv in argvs if argv[0] == "inspect" and F_APP_SETTINGS in argv]
    assert not [argv for argv in argvs if is_id_probe(argv)]


# --- the fake itself: it enforces the exec user, the boot ownership and the owner-only rules


def docker_by_hand(host: Host, *args: str) -> subprocess.CompletedProcess:
    env = {"PATH": f"{host.root / 'bin'}:/usr/bin:/bin", "FAKE_HOST_ROOT": str(host.root),
           "HOME": str(host.root / "home"), "PYTHONDONTWRITEBYTECODE": "1"}
    return subprocess.run([str(host.root / "bin/docker"), *args], env=env, capture_output=True, text=True)


def own_volume(host: Host, volume: str, who: tuple[int, int], mode: int | None = None) -> None:
    table = host.owners()
    table[volume] = list(who)
    (host.root / "owners.json").write_text(json.dumps(table))
    if mode is not None:
        host.vol(volume).chmod(mode)


def issue_by_hand(host: Host, user: str) -> subprocess.CompletedProcess:
    return docker_by_hand(host, "exec", "--user", user, DASH_C, "hermes", "kanban-workspace-token", "issue",
                          "--surface", "workspace", "--out", "/tokens/workspace-new")


def test_the_fake_issues_a_token_for_the_user_that_owns_the_folder(host: Host):
    own_volume(host, TOKENS_V, (10000, 10000), 0o700)
    result = issue_by_hand(host, "10000:10000")
    assert result.returncode == 0, result.stdout + result.stderr
    made = host.vol(TOKENS_V) / "workspace-new"
    assert stat.S_IMODE(made.stat().st_mode) == 0o600 and host.owner_of(made) == (10000, 10000)


@pytest.mark.parametrize("user, owner, mode, message", [
    ("1000:1000", (1000, 1000), 0o700, "Permission denied"),
    ("10000:10000", (1000, 1000), 0o700, "is not owned by the current user"),
    ("10000:10000", (10000, 10000), 0o750, "is not owner-only"),
    ("1000", (1000, 0), 0o700, "Permission denied"),
], ids=["home-of-another-user", "folder-of-another-user", "folder-not-owner-only", "uid-only"])
def test_the_fake_refuses_a_token_for_the_wrong_user_like_the_real_command(host: Host, user, owner, mode, message):
    own_volume(host, TOKENS_V, owner, mode)
    result = issue_by_hand(host, user)
    assert result.returncode == 2 and result.stdout.startswith("error: ") and message in result.stdout
    assert not (host.vol(TOKENS_V) / "workspace-new").exists()


def test_the_fake_stops_a_writer_that_runs_as_a_user_without_access(host: Host):
    result = docker_by_hand(host, "exec", "--user", "1000:1000", DASH_C, "sh", "-c", 'printf x >> "$1"', "sh",
                            "/opt/data/.env")
    assert result.returncode == 2 and "Permission denied" in result.stderr
    assert not (host.vol(HOME_V) / ".env").exists()


def test_the_fake_boot_hook_gives_the_listed_parts_of_the_home_to_the_runtime_account(host: Host):
    home = host.vol(HOME_V)
    (home / "profiles/p").mkdir(parents=True)
    for name in (".env", "config.yaml", "profiles/p/.env"):
        (home / name).write_text("x\n")
    table = host.owners()
    for path in [home, *home.rglob("*")]:
        table[str(path.relative_to(host.root / "volumes"))] = [1000, 1000]
    (host.root / "owners.json").write_text(json.dumps(table))
    assert docker_by_hand(host, "start", DASH_C).returncode == 0
    for name in ("", ".env", "config.yaml", "profiles", "profiles/p", "profiles/p/.env"):
        assert host.owner_of(home / name) == (10000, 10000), name
    assert host.owner_of(home / "keep.txt") == (1000, 1000)


def test_reset_names_only_template_containers_and_volumes(shared_reset: SharedReset):
    host = shared_reset.host
    template = parse_template()
    allowed = set(template["containers"]) | set(template["volumes"])
    calls = host.calls("docker")
    found = set().union(*(docker_names(c) for c in calls))
    assert found <= allowed
    assert not found & (set(DECOY_CONTAINERS) | set(DECOY_VOLUMES) | {PROD_C})
    for call in calls:
        argv = call["argv"]
        assert argv[0] in ("inspect", "start", "restart", "stop", "exec", "run")
        assert not {"-v", "--volume", "--volumes-from"} & set(argv)
        assert not [part for part in argv if f"source={PROXY_V}" in part]
    proxy_calls = {docker_target(c)[0] for c in calls if docker_target(c)[1] == PROXY_C}
    assert proxy_calls == {"inspect", "start"}, "reset may only look at and start the proxy container"


def test_reset_scripts_carry_no_container_or_volume_names():
    # The one name that is not a docker resource: the reset spells the anchor slug once, as a plain assignment.
    anchor_assignment = re.compile(r"""\s*[a-z_]+=["']?%s["']?\s*""" % re.escape(ANCHOR))
    for script in (RESET, CHECK):
        for number, line in enumerate(script.read_text().splitlines(), 1):
            if not line.lstrip().startswith("#") and not (script == RESET and anchor_assignment.fullmatch(line)):
                assert not resource_names_in(line), f"{script.name}:{number} hard-codes a name"


def writer_calls(host: Host) -> list[dict]:
    return [c for c in host.calls("docker") if docker_target(c)[0] == "exec" and "--user" in c["argv"]]


def test_reset_polls_every_two_seconds_until_healthy(host: Host):
    host.update(lambda state: state["containers"][DASH_C].update(health_delay=3))
    reset_ok(host)
    polls = [c["argv"] for c in host.calls("sleep")]
    assert 3 <= len(polls) <= 60 and all(argv == ["2"] for argv in polls)


@pytest.mark.parametrize("who, trouble", [
    (DASH_C, {"health": "unhealthy"}),
    (DASH_C, {"health_delay": 10**6}),
    (GATE_C, {"stays_down": True}),
    (PROXY_C, {"stays_down": True, "running": False}),
    (BROWSER_C, {"stays_down": True, "running": False}),
], ids=["unhealthy", "never-healthy", "gateway-stays-down", "proxy-stays-down", "browser-stays-down"])
def test_reset_gives_up_after_120_seconds_naming_the_container(host: Host, who: str, trouble: dict):
    host.update(lambda state: state["containers"][who].update(trouble))
    result = run_reset(host)
    assert result.returncode != 0
    assert who in all_output(result)
    assert 55 <= len(host.calls("sleep")) <= 60, "120 seconds in total, polled every 2 seconds"


def set_healthcheck(who: str, test):
    def mutate(state: dict) -> None:
        if test == "absent":
            state["containers"][who].pop("healthcheck", None)
        else:
            state["containers"][who]["healthcheck"] = test
    return mutate


@pytest.mark.parametrize("who", STACK_CONTAINERS)
def test_reset_refuses_a_container_without_a_health_check_before_it_changes_anything(host: Host, who: str):
    host.update(set_healthcheck(who, "absent"))
    before = snapshot(host)
    result = run_reset(host)
    assert result.returncode != 0
    assert who in result.stderr and "health check" in result.stderr
    assert "must create it with a health check" in result.stderr and "nothing was changed" in result.stderr
    assert_nothing_changed(host, before)
    assert host.calls("sleep") == []


@pytest.mark.parametrize("test, how", [(["NONE"], "kind-none"), ([], "unreadable")])
def test_reset_refuses_a_health_check_that_is_none_or_cannot_be_read(host: Host, test: list, how: str):
    host.update(set_healthcheck(PROXY_C, test))
    before = snapshot(host)
    result = run_reset(host)
    assert result.returncode != 0
    assert PROXY_C in result.stderr and "must create it with a health check" in result.stderr
    assert_nothing_changed(host, before)


def test_reset_reads_the_health_check_kind_of_all_five_containers_before_it_changes_anything(shared_reset: SharedReset):
    calls = shared_reset.host.calls("docker")
    reads = [i for i, c in enumerate(calls) if c["argv"][0] == "inspect" and F_HEALTH_KIND in c["argv"]]
    assert sorted(calls[i]["argv"][-1] for i in reads) == sorted(STACK_CONTAINERS)
    changing = [i for i, c in enumerate(calls)
                if docker_target(c)[0] not in ("inspect", "exec") or "-i" in c["argv"] or "--user" in c["argv"]]
    assert changing and max(reads) < min(changing)


def test_reset_invents_no_probe_of_its_own(shared_reset: SharedReset):
    docker = shared_reset.host.calls("docker")
    assert shared_reset.host.calls("curl") == []
    entered = {docker_target(c)[1] for c in docker if docker_target(c)[0] == "exec"}
    assert entered <= {APP_C, DASH_C}, "no command is run in the proxy, the gateway or the browser to probe it"
    words = {word for c in docker for word in c["argv"]}
    assert not words & {"curl", "wget", "nc", "ncat", "telnet", "ping"}
    assert not [word for word in words if "://" in word]


class PendingReset(NamedTuple):
    host: Host
    result: subprocess.CompletedProcess


@pytest.fixture(scope="module")
def pending_reset(tmp_path_factory) -> PendingReset:
    """One reset in which four containers never answer: two starting, one without a status, one unhealthy."""
    first = Host(tmp_path_factory.mktemp("pending") / "host")

    def never_answer(state):
        for who in (APP_C, BROWSER_C):
            state["containers"][who].update(health_delay=10**6, starting_polls=10**6)
        state["containers"][GATE_C].pop("health")
        state["containers"][PROXY_C].update(health="unhealthy")

    first.update(never_answer)
    return PendingReset(first, run_reset(first))


def pending_names(shared: PendingReset) -> str:
    """The containers that the refusal names after its budget of 120 seconds."""
    assert shared.result.returncode != 0
    assert "120 seconds" in shared.result.stderr
    return shared.result.stderr.split("120 seconds", 1)[-1]


@pytest.mark.parametrize("who", [APP_C, BROWSER_C])
def test_reset_does_not_call_a_container_ready_while_its_health_is_starting(pending_reset: PendingReset, who: str):
    assert who in pending_names(pending_reset)


def test_reset_does_not_call_a_container_ready_when_its_health_status_is_missing(pending_reset: PendingReset):
    assert GATE_C in pending_names(pending_reset)


def test_reset_does_not_call_a_container_ready_while_its_health_is_unhealthy(pending_reset: PendingReset):
    assert PROXY_C in pending_names(pending_reset)


def test_reset_names_no_container_that_answered_when_it_gives_up(pending_reset: PendingReset):
    assert DASH_C not in pending_names(pending_reset)


def test_reset_gives_pending_containers_120_seconds_in_total_polled_every_2_seconds(pending_reset: PendingReset):
    assert 55 <= len(pending_reset.host.calls("sleep")) <= 60


@pytest.mark.parametrize("who", [GATE_C, PROXY_C, BROWSER_C])
def test_reset_fails_naming_a_container_that_cannot_start(host: Host, who: str):
    host.update(lambda state: state["containers"][who].update(running=False, fails_to_start=True))
    result = run_reset(host)
    assert result.returncode != 0
    assert who in all_output(result)


@pytest.mark.parametrize("how", ["unset", "missing", "empty", "blank"])
def test_reset_skips_the_model_token_with_one_value_free_line(host: Host, how: str):
    env: dict = {}
    if how == "missing":
        env["JOURNEY_MODEL_TOKEN_FILE"] = str(host.root / "no-such-token-file")
    elif how in ("empty", "blank"):
        env["JOURNEY_MODEL_TOKEN_FILE"] = str(host.model_token_file("" if how == "empty" else " \n\n"))
    result = run_reset(host, token=False, **env)
    assert result.returncode == 0, result.stdout + result.stderr
    for path in home_files(host).values():
        assert "CLAUDE_CODE_OAUTH_TOKEN" not in path.read_text()
    skips = [line for line in all_output(result).splitlines() if "skip" in line.lower()]
    assert len(skips) == 1 and "JOURNEY_MODEL_TOKEN_FILE" in skips[0]
    if env:
        assert env["JOURNEY_MODEL_TOKEN_FILE"] not in all_output(result)
    # the anchor Project and its board, 5 tokens and 2 API server keys; no model token writes
    assert len(writer_calls(host)) == ANCHOR_WRITES + 7


def test_reset_stops_when_a_token_cannot_be_issued(host: Host):
    host.update(lambda state: state["hermes"].update(issue_fail="connections"))
    result = run_reset(host)
    assert result.returncode != 0
    assert "connections" in all_output(result)
    done = [docker_target(c) for c in host.calls("docker")]
    assert not [step for step in done if step[0] == "restart"]
    assert ("start", GATE_C) not in done
    assert_secret_free(host, result, [MODEL_TOKEN])


def test_reset_prints_no_skip_line_when_the_model_token_is_written(shared_reset: SharedReset):
    assert "skip" not in all_output(shared_reset.result).lower()
    assert len(writer_calls(shared_reset.host)) == ANCHOR_WRITES + 9


def test_reset_leaves_the_repository_files_alone(shared_reset: SharedReset):
    assert sha_tree(STACK_HOME) == shared_reset.repo_before


# ------------------------------------------------------------------ reset_stack.sh: a stack that was just created

APP_USER = "svc-sentinel"  # the owner app is created with a user name; only its image says which numbers that is
APP_IDS = (4321, 4322)
SETTING_SENTINELS = {
    "RAPHAEL_ACCESS_ISSUER": "https://setting-sentinel-issuer.example.invalid/issuer",
    "RAPHAEL_OWNER_MEMBERSHIP_JSON": '{"owner": "setting-sentinel-membership"}',
    "NODE_EXTRA_CA_CERTS": "/etc/setting-sentinel/ca.pem",
}
NAME_MARK = "odd-name-marker"
CUSTOM_NAMES = dict(zip(TOKEN_VARS, ["WS_1.tok", "rec-2", "Conn.3", "auto_4", "models-5.tok", "exec_6", "plan.7-x"]))


def sentinel_settings(state: dict) -> None:
    for name, value in SETTING_SENTINELS.items():
        set_app_env(name, value)(state)


def fresh_stack(host: Host) -> None:
    """The stack as its creation leaves it: owner app and gateway never ran, the token folder is empty and root's."""
    for path in host.vol(TOKENS_V).iterdir():
        path.unlink()
    host.set_identities(APP_IDS, (10000, 10000))
    own_volume(host, TOKENS_V, (0, 0))

    def mutate(state: dict) -> None:
        containers = state["containers"]
        state["images"][containers[APP_C]["image"]]["users"] = {APP_USER: list(APP_IDS)}
        containers[APP_C].update(config_user=APP_USER, exits_without_token_files=True)
        for who in (APP_C, GATE_C):
            containers[who].pop("health")  # a container that never ran has no health state
            containers[who].update(running=False, health_on_start="healthy", health_delay=2)
        sentinel_settings(state)
    host.update(mutate)


def token_setting(var: str, *values: str):
    """The owner app holds the setting once per value (no value leaves it unset)."""
    def mutate(state: dict) -> None:
        box = state["containers"][APP_C]
        box["env"] = [item for item in box["env"] if not item.startswith(var + "=")] + [f"{var}={v}" for v in values]
    return mutate


def settings_to_hide(host: Host) -> list[str]:
    """The values of the owner app's token file settings and sentinel settings, as the container holds them now."""
    pairs = [item.partition("=") for item in host.state()["containers"][APP_C]["env"]]
    return [value for name, _, value in pairs if name in (*TOKEN_VARS, *SETTING_SENTINELS) and value]


def assert_no_setting_values(host: Host, result: subprocess.CompletedProcess, values: list[str]) -> None:
    """No output line and no recorded command line carries a setting value; only the inspect itself returns them."""
    assert values and all(len(value) >= 8 for value in values)
    answers = "\n".join(c.get("stdout", "") + c.get("stderr", "") for c in host.calls("docker")
                        if F_APP_SETTINGS not in c["argv"])
    for value in values:
        assert value not in all_output(result), "a setting value was printed"
        assert value not in argv_text(host), "a setting value reached a command line"
        assert value not in answers, "a container command printed a setting value"


def assert_user_only_on_the_probe(host: Host, result: subprocess.CompletedProcess) -> None:
    """The configured user is the one setting value that must be on a command line: --user of the id run."""
    carrying = [c["argv"] for c in host.calls() if APP_USER in "\x00".join(c["argv"])]
    probes = [c["argv"] for c in host.calls("docker") if is_id_probe(c["argv"])]
    assert carrying == probes and len(probes) <= 1
    assert all(argv[argv.index("--user") + 1] == APP_USER for argv in probes)
    assert APP_USER not in all_output(result)
    assert APP_USER not in "\n".join(c.get("stdout", "") + c.get("stderr", "") for c in host.calls("docker")
                                     if F_APP_SETTINGS not in c["argv"])


def change_steps(host: Host) -> list[tuple[str, str | None]]:
    """The changing docker calls of a reset in order: stops, starts, restarts, helper runs by volume, writes."""
    steps = []
    for call in host.calls("docker"):
        sub, target = docker_target(call)
        if sub in ("stop", "start", "restart"):
            steps.append((sub, target))
        elif sub == "run":
            steps.append(("helper", mount_source(call)))
        elif sub == "exec" and "--user" in call["argv"]:
            steps.append(("write", target))
    return steps


def test_reset_starts_a_stopped_owner_app_on_a_stack_that_was_just_created(host: Host):
    fresh_stack(host)
    image = host.state()["containers"][APP_C]["image"]
    result = reset_ok(host)

    containers = host.state()["containers"]
    for who in STACK_CONTAINERS:
        assert (containers[who]["running"], containers[who].get("health")) == (True, "healthy"), who
    assert "the stack is reset and every container answers" in result.stdout
    assert len(host.calls("sleep")) >= 2, "the two containers that were down report starting before they answer"

    tokens = host.vol(TOKENS_V)
    names = set(SURFACE_ENV) | {"executor", "planner"}
    assert {p.name for p in tokens.iterdir()} == names
    assert host.owner_of(tokens) == APP_IDS and stat.S_IMODE(tokens.stat().st_mode) == 0o700
    for name in names:
        info = (tokens / name).lstat()
        assert stat.S_ISREG(info.st_mode) and stat.S_IMODE(info.st_mode) == 0o600 and 0 < info.st_size <= 16 * 1024
        assert host.owner_of(tokens / name) == APP_IDS
    for surface in SURFACE_ENV:
        assert (tokens / surface).read_text().startswith(f"tok-{surface}-")
    for name, path in zip(("executor", "planner"), home_files(host).values()):
        assert (tokens / name).read_text().strip() == parse_env_file(path)[1]["API_SERVER_KEY"]

    steps = change_steps(host)  # the order of a reset of a running stack
    assert set(steps[:2]) == {("stop", GATE_C), ("stop", DASH_C)}
    assert set(steps[2:4]) == {("helper", HOME_V), ("helper", TOKENS_V)}
    assert steps[4] == ("start", DASH_C)
    assert steps[5:-6] == [("write", DASH_C)] * (ANCHOR_WRITES + 9)
    assert steps[-6] == ("helper", TOKENS_V)
    assert steps[-5:-2] == [("restart", APP_C), ("restart", DASH_C), ("start", GATE_C)]
    assert set(steps[-2:]) == {("start", PROXY_C), ("start", BROWSER_C)}

    calls = host.calls("docker")  # the owner app is read without running it
    assert [c["argv"] for c in calls if c["argv"][0] == "inspect" and F_APP_SETTINGS in c["argv"]] == [
        ["inspect", "--type", "container", "--format", F_APP_SETTINGS, APP_C]]
    assert [c["argv"] for c in calls if is_id_probe(c["argv"])] == [
        ["run", "--rm", "--network", "none", "--user", APP_USER, "--entrypoint", "sh", image, "-c", ID_SCRIPT]]
    assert not [c for c in calls if docker_target(c) == ("exec", APP_C)]
    kinds = [docker_target(c)[0] for c in calls]
    assert kinds.index("id-run") < kinds.index("stop")

    assert_secret_free(host, result, generated_secrets(host))
    assert_no_setting_values(host, result, settings_to_hide(host))
    assert_user_only_on_the_probe(host, result)


@pytest.mark.parametrize("app", ["running", "stopped"])
def test_reset_still_refuses_a_stopped_dashboard_and_changes_nothing(host: Host, app: str):
    if app == "stopped":
        fresh_stack(host)
    else:
        host.update(sentinel_settings)
    host.update(lambda state: state["containers"][DASH_C].update(running=False))
    before = snapshot(host)
    result = run_reset(host)
    assert result.returncode != 0
    assert DASH_C in result.stderr and "nothing was changed" in result.stderr
    assert_nothing_changed(host, before)
    assert host.calls("sleep") == []
    assert_secret_free(host, result, [MODEL_TOKEN])
    assert_no_setting_values(host, result, settings_to_hide(host))
    assert_user_only_on_the_probe(host, result)


def test_reset_uses_the_token_file_names_that_the_stopped_owner_app_settings_give(host: Host):
    fresh_stack(host)

    def custom_names(state: dict) -> None:
        for var, name in CUSTOM_NAMES.items():
            token_setting(var, f"/run/tokens/{name}")(state)
    host.update(custom_names)
    result = reset_ok(host)

    tokens = host.vol(TOKENS_V)
    assert {p.name for p in tokens.iterdir()} == set(CUSTOM_NAMES.values())
    for surface, var in SURFACE_ENV.items():
        assert (tokens / CUSTOM_NAMES[var]).read_text().startswith(f"tok-{surface}-")
    for var, path in zip(KEY_ENV.values(), home_files(host).values()):
        assert (tokens / CUSTOM_NAMES[var]).read_text().strip() == parse_env_file(path)[1]["API_SERVER_KEY"]
    issued = [c["argv"][c["argv"].index("--out") + 1] for c in writer_calls(host) if "--out" in c["argv"]]
    assert sorted(issued) == sorted(f"/tokens/{CUSTOM_NAMES[var]}" for var in SURFACE_ENV.values())
    containers = host.state()["containers"]
    assert (containers[APP_C]["running"], containers[APP_C]["health"]) == (True, "healthy")

    assert_secret_free(host, result, generated_secrets(host))
    assert_no_setting_values(host, result, settings_to_hide(host))
    assert_user_only_on_the_probe(host, result)


BAD_TOKEN_NAMES = {
    "shared-with-a-surface": (KEY_ENV["default"], ["/run/tokens/workspace"]),
    "shared-by-two-surfaces": (SURFACE_ENV["models"], ["/run/tokens/recommendations"]),
    "shared-by-the-two-keys": (KEY_ENV["planner"], ["/run/tokens/executor"]),
    "dot-dot": (SURFACE_ENV["workspace"], ["/run/tokens/.."]),
    "parent-folder": (SURFACE_ENV["workspace"], [f"/run/tokens/../{NAME_MARK}"]),
    "nested": (SURFACE_ENV["recommendations"], [f"/run/tokens/{NAME_MARK}/file"]),
    "empty": (SURFACE_ENV["connections"], ["/run/tokens/"]),
    "hidden": (SURFACE_ENV["automations"], [f"/run/tokens/.{NAME_MARK}"]),
    "leading-dash": (SURFACE_ENV["models"], [f"/run/tokens/-{NAME_MARK}"]),
    "space": (KEY_ENV["default"], [f"/run/tokens/{NAME_MARK} name"]),
    "command-substitution": (KEY_ENV["planner"], [f"/run/tokens/$({NAME_MARK})"]),
    "double-quote": (SURFACE_ENV["workspace"], [f'/run/tokens/{NAME_MARK}"quote']),
    "backslash": (SURFACE_ENV["workspace"], [f"/run/tokens/{NAME_MARK}\\back"]),
    "tab": (SURFACE_ENV["workspace"], [f"/run/tokens/{NAME_MARK}\ttab"]),
    "newline": (SURFACE_ENV["workspace"], [f"/run/tokens/{NAME_MARK}\nHERMES_API_TOKEN_FILE=/run/tokens/forged"]),
    "unset": (SURFACE_ENV["models"], []),
    "set-twice": (SURFACE_ENV["models"], [f"/run/tokens/{NAME_MARK}-a", f"/run/tokens/{NAME_MARK}-b"]),
}


@pytest.mark.parametrize("case", sorted(BAD_TOKEN_NAMES))
def test_reset_refuses_a_shared_or_not_plain_token_file_name_of_the_stopped_owner_app(host: Host, case: str):
    var, values = BAD_TOKEN_NAMES[case]
    fresh_stack(host)
    host.update(token_setting(var, *values))
    before = snapshot(host)
    result = run_reset(host)
    assert result.returncode != 0
    assert var in result.stderr and APP_C in result.stderr and "nothing was changed" in result.stderr
    assert_nothing_changed(host, before)
    assert host.calls("sleep") == []
    assert_secret_free(host, result, [MODEL_TOKEN])
    assert_no_setting_values(host, result, settings_to_hide(host) + [NAME_MARK])
    assert_user_only_on_the_probe(host, result)


# ------------------------------------------------------------------ reset_stack.sh: the identity probe fails closed

EXTRA_ID = "4999"
UID_TEXT, GID_TEXT = (str(number) for number in APP_IDS)
# (printed text, exit status) of `id -u`, then of `id -g`; None prints nothing.
BAD_PROBES = {
    "first-read-fails": ((UID_TEXT, 23), (GID_TEXT, 0)),
    "second-read-fails": ((UID_TEXT, 0), (GID_TEXT, 23)),
    "extra-line-from-id-u": ((f"{UID_TEXT}\n{EXTRA_ID}", 0), (GID_TEXT, 0)),
    "extra-line-from-id-g": ((UID_TEXT, 0), (f"{GID_TEXT}\n{EXTRA_ID}", 0)),
    "blank-line-between": ((UID_TEXT + "\n", 0), (GID_TEXT, 0)),
    "trailing-blank-line": ((UID_TEXT, 0), (GID_TEXT + "\n", 0)),
    "one-line": ((UID_TEXT, 0), (None, 0)),
    "no-output": ((None, 0), (None, 0)),
}


def probe_says(reads):
    def mutate(state: dict) -> None:
        box = state["containers"][APP_C]
        for key, (text, status) in zip(("id -u", "id -g"), reads):
            box.setdefault("answers", {})[key] = text
            box.setdefault("exit_codes", {})[key] = status
    return mutate


@pytest.mark.parametrize("case", list(BAD_PROBES))
def test_reset_refuses_an_identity_probe_unless_both_reads_succeed_and_print_exactly_two_numbers(
        host: Host, case: str):
    fresh_stack(host)
    host.update(probe_says(BAD_PROBES[case]))
    before = snapshot(host)
    result = run_reset(host)
    assert result.returncode != 0, "the reset went on after an identity probe that has to be refused"
    runs = [c["argv"] for c in host.calls("docker") if c["argv"][0] == "run"]
    assert [argv[-2:] for argv in runs] == [["-c", "id -u && id -g"]], "the identity probe is not `id -u && id -g`"
    assert APP_C in result.stderr and "nothing was changed" in result.stderr
    assert_nothing_changed(host, before)
    assert host.calls("sleep") == []
    assert_secret_free(host, result, [MODEL_TOKEN])
    assert_no_setting_values(host, result, settings_to_hide(host))
    assert_user_only_on_the_probe(host, result)
    for number in (UID_TEXT, GID_TEXT, EXTRA_ID):  # not even a helper number is printed or put on a command line
        assert number not in all_output(result), "a helper number was printed"
        assert number not in argv_text(host), "a helper number reached a command line"


def test_the_fake_runs_the_probe_text_in_a_real_shell_with_a_stand_in_id(host: Host):
    image = host.state()["containers"][APP_C]["image"]

    def probe(text: str) -> tuple[int, str]:
        done = docker_by_hand(host, "run", "--rm", "--network", "none", "--user", "1000:1000", "--entrypoint", "sh",
                              image, "-c", text)
        return done.returncode, done.stdout
    assert probe("id -u && id -g") == (0, "1000\n1000\n")
    host.update(probe_says(((UID_TEXT, 23), (GID_TEXT, 0))))  # `&&` stops after a failed first read, `;` goes on
    assert probe("id -u && id -g") == (23, f"{UID_TEXT}\n")
    assert probe("id -u; id -g") == (0, f"{UID_TEXT}\n{GID_TEXT}\n")
    assert probe("id -u; touch somewhere")[0] == 97
    host.update(probe_says(((f"{UID_TEXT}\n{EXTRA_ID}", 0), (GID_TEXT + "\n", 0))))
    assert probe("id -u && id -g") == (0, f"{UID_TEXT}\n{EXTRA_ID}\n{GID_TEXT}\n\n")
    host.update(probe_says(((None, 0), (None, 0))))
    assert probe("id -u && id -g") == (0, "")


# ------------------------------------------------------------------ isolation_check.sh


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


def proxy_issuer() -> str:
    match = re.search(r'^ISSUER = "([^"]*)"$', PROXY_SOURCE.read_text(), re.M)
    assert match, "synthetic_proxy.py no longer has an ISSUER constant line"
    return match.group(1)


def b64d(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def profile_folder(host: Host, profile: str) -> Path:
    home = host.vol(HOME_V)
    return home if profile == "default" else home / "profiles" / f"raphael-{profile}"


def edit_config(host: Host, profile: str, change) -> None:
    path = profile_folder(host, profile) / "config.yaml"
    data = yaml.safe_load(path.read_text()) or {}
    change(data)
    path.write_text(yaml.safe_dump(data))


def add_env_line(host: Host, profile: str, line: str) -> None:
    with open(profile_folder(host, profile) / ".env", "a") as handle:
        handle.write(line + "\n")


def extra_env_name(who: str):
    return lambda host: host.update(lambda s: s["containers"][who]["env"].append("STRAY_SETTING=1"))


def extra_mount(who: str, entry: dict):
    return lambda host: host.update(lambda s: s["containers"][who]["mounts"].append(entry))


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


def curl_gives(**settings):
    return lambda host: host.update(lambda s: s["curl"].update(settings))


# One thing broken per case: (the check that must fail, how to break it, script environment changes).
BREAKS = {
    "extra-env-name-in-app": ("names-on-template", extra_env_name(APP_C), {}),
    "extra-env-name-in-browser": ("names-on-template", extra_env_name(BROWSER_C), {}),
    "extra-env-name-in-gateway": ("names-on-template", extra_env_name(GATE_C), {}),
    "bind-mount": ("names-on-template",
                   extra_mount(APP_C, {"type": "bind", "name": "", "dest": "/host-dir"}), {}),
    "foreign-volume": ("names-on-template",
                       extra_mount(GATE_C, {"type": "volume", "name": DECOY_VOLUMES[0], "dest": "/old"}), {}),
    "home-env-name-default": ("names-on-template", lambda h: add_env_line(h, "default", "STRAY_SETTING=1"), {}),
    "home-env-name-planner": ("names-on-template", lambda h: add_env_line(h, "planner", "STRAY_SETTING=1"), {}),
    "home-env-name-designer": ("names-on-template", lambda h: add_env_line(h, "designer", "STRAY_SETTING=1"), {}),
    "home-env-line-without-name": ("names-on-template", lambda h: add_env_line(h, "planner", "just-text"), {}),
    "home-env-line-without-name-designer": (
        "names-on-template", lambda h: add_env_line(h, "designer", "just-text"), {}),
    "dispatcher-on-default": ("dispatcher-off", dispatcher_on("default"), {}),
    "dispatcher-on-planner": ("dispatcher-off", dispatcher_on("planner"), {}),
    "dispatcher-on-designer": ("dispatcher-off", dispatcher_on("designer"), {}),
    "dispatcher-unset-planner": ("dispatcher-off", dispatcher_unset("planner"), {}),
    "dispatcher-unset-designer": ("dispatcher-off", dispatcher_unset("designer"), {}),
    "cron-job-default": ("cron-lists-empty", cron_job("default"), {}),
    "cron-job-planner": ("cron-lists-empty", cron_job("planner"), {}),
    "cron-job-designer": ("cron-lists-empty", cron_job("designer"), {}),
    "cron-command-fails": ("cron-lists-empty", hermes_gives(cron_rc=1), {}),
    "platform-in-default-config": ("no-messaging-platform", platform_enabled("default"), {}),
    "platform-in-planner-config": ("no-messaging-platform", platform_enabled("planner"), {}),
    "platform-in-designer-config": ("no-messaging-platform", platform_enabled("designer"), {}),
    "message-delivered": ("telegram-send-fails", hermes_gives(send_rc=0), {}),
    "production-answers-200": ("production-get-refused", curl_gives(code=200), {}),
    "production-answers-404": ("production-get-refused", curl_gives(code=404), {}),
    "production-answers-500": ("production-get-refused", curl_gives(code=500), {}),
    "redirect-to-the-same-host": ("production-get-refused",
                                  curl_gives(code=302, redirect=SAME_HOST_REDIRECT), {}),
    "redirect-to-the-same-host-other-case-port-user": (
        "production-get-refused",
        curl_gives(code=302, redirect="https://someone@PROD.example.invalid:8443/x"), {}),
    "redirect-to-the-same-host-trailing-dot": (
        "production-get-refused", curl_gives(code=302, redirect="https://prod.example.invalid./x"), {}),
    "redirect-without-location": ("production-get-refused", curl_gives(code=302, redirect=""), {}),
    "redirect-without-a-host": ("production-get-refused", curl_gives(code=302, redirect="/login"), {}),
    "curl-fails": ("production-get-refused", curl_gives(rc=28), {}),
    "origin-unset": ("production-get-refused", None, {"JOURNEY_PRODUCTION_ORIGIN": None}),
    "origin-empty": ("production-get-refused", None, {"JOURNEY_PRODUCTION_ORIGIN": ""}),
}


@pytest.mark.parametrize("case", sorted(BREAKS))
def test_check_reports_a_broken_check_and_still_runs_the_others(stack: Host, case: str):
    broken, mutate, env = BREAKS[case]
    if mutate:
        mutate(stack)
    before = snapshot(stack)
    result = run_check(stack, **env)
    assert result.returncode == 1, result.stdout + result.stderr
    assert list(verdicts(result)) == CHECKS, "one line per check, in order"
    assert failed_checks(result) == {broken}
    assert snapshot(stack) == before
    for value in (STACK_HOST, PROD_ORIGIN, SAME_HOST_REDIRECT, SIGN_IN_REDIRECT, MODEL_TOKEN):
        assert value not in all_output(result)


@pytest.mark.parametrize("code, redirect", [
    (401, ""), (403, ""), (302, SIGN_IN_REDIRECT), (307, SIGN_IN_REDIRECT),
    (303, "https://other.example.invalid/"),
])
def test_check_accepts_a_refusal_or_a_redirect_elsewhere(stack: Host, code: int, redirect: str):
    stack.update(lambda s: s["curl"].update(code=code, redirect=redirect))
    result = run_check(stack)
    assert result.returncode == 0, result.stdout
    assert verdicts(result)["production-get-refused"] == "PASS"


@pytest.mark.parametrize("origin", [
    "prod.example.invalid", "ftp://prod.example.invalid", "https://", 'https://prod.example.invalid/"x',
    'https://prod.example.invalid/\nheader = "x: y"', "https://prod.example.invalid/ -o /etc/x",
])
def test_check_refuses_an_origin_that_is_not_a_plain_address_without_calling_curl(stack: Host, origin: str):
    result = run_check(stack, JOURNEY_PRODUCTION_ORIGIN=origin)
    assert result.returncode == 1
    assert failed_checks(result) == {"production-get-refused"}
    assert stack.calls("curl") == []


@pytest.mark.parametrize("args", [[], ["bogus"], [""], ["SERVER2"], ["server2", "server1"]])
def test_check_prints_usage_and_exits_2_for_a_missing_or_other_argument(host: Host, args: list):
    result = host.run(CHECK, *args)
    assert result.returncode == 2
    assert "usage" in result.stderr.lower() and result.stdout == ""
    assert host.calls() == []


@pytest.mark.parametrize("stack_host", ["some-other-host.invalid", "", None])
def test_check_server2_refuses_the_wrong_host_before_any_other_check(host: Host, stack_host):
    result = host.run(CHECK, "server2", env={"JOURNEY_STACK_HOST": stack_host})
    assert result.returncode == 1
    lines = result.stdout.splitlines()
    assert len(lines) == 1 and lines[0].startswith("FAIL host")
    assert host.calls("docker") == [] and host.calls("curl") == []
    assert STACK_HOST not in all_output(result)


def test_check_server2_passes_every_check_on_a_clean_stack(clean_check: CleanCheck):
    stack, before, result = clean_check
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.splitlines() == [f"PASS {name}" for name in CHECKS]
    assert result.stderr == ""
    assert snapshot(stack) == before


def test_check_server2_names_only_template_entries(clean_check: CleanCheck):
    stack, _, result = clean_check
    assert result.returncode == 0, result.stdout
    template = parse_template()
    allowed = set(template["containers"]) | set(template["volumes"])
    docker = stack.calls("docker")
    assert {docker_target(c)[0] for c in docker} <= {"inspect", "exec"}
    found = set().union(*(resource_names_in(part) for c in docker for part in c["argv"]))
    assert found <= allowed
    assert not found & (set(DECOY_CONTAINERS) | set(DECOY_VOLUMES) | {PROD_C})
    images = {entry["image"] for name, entry in stack.state()["containers"].items() if name in allowed}
    for call in docker:
        if "--type" in call["argv"] and call["argv"][call["argv"].index("--type") + 1] == "image":
            assert call["argv"][-1] in images, "an image id that no template container uses"


def test_check_reads_the_config_with_the_interpreter_of_the_hermes_command(clean_check: CleanCheck):
    uses = (clean_check.host.root / "interp.log").read_text().splitlines()
    assert uses.count("-") >= 5, "the served-profile, dispatcher and platform reads use that interpreter"


def test_check_reads_the_configured_hermes_home_of_the_dashboard_and_the_gateway(clean_check: CleanCheck):
    calls = clean_check.host.calls("docker")
    read = [c["argv"][-1] for c in calls if c["argv"][0] == "inspect" and F_HERMES_HOME in c["argv"]]
    assert sorted(read) == sorted([DASH_C, GATE_C])


@pytest.mark.parametrize("value", ["/somewhere/else", "/opt/data/"])
@pytest.mark.parametrize("who", [DASH_C, GATE_C])
def test_check_fails_names_when_hermes_home_is_not_the_home_mount(stack: Host, who: str, value: str):
    stack.update(set_hermes_home(who, value))
    result = run_check(stack)
    assert result.returncode == 1, result.stdout
    assert list(verdicts(result)) == CHECKS
    line = next(line for line in result.stdout.splitlines() if line.startswith("FAIL names-on-template"))
    assert who in line
    if who == DASH_C:
        assert failed_checks(result) == {"names-on-template"}
    assert "names-on-template" in failed_checks(result)


def test_check_fails_names_when_the_gateway_uses_another_directory_with_a_stale_state(stack: Host):
    other = stack.root / "gateway-own-filesystem" / "data"
    shutil.copytree(stack.vol(HOME_V), other)
    (other / "stale-state.db").write_text("sentinel rows that must stay\n")
    stack.update(set_hermes_home(GATE_C, str(other)))
    before = sha_tree(other)
    result = run_check(stack)
    assert result.returncode == 1, result.stdout
    assert list(verdicts(result)) == CHECKS
    assert failed_checks(result) == {"names-on-template"}
    assert GATE_C in next(line for line in result.stdout.splitlines() if line.startswith("FAIL names-on-template"))
    assert sha_tree(other) == before


def test_check_fails_closed_when_the_gateway_is_not_running(stack: Host):
    stack.update(lambda s: s["containers"][GATE_C].update(running=False))
    result = run_check(stack)
    assert result.returncode == 1
    assert list(verdicts(result)) == CHECKS
    assert failed_checks(result) == set(CHECKS) - {"production-get-refused"}


@pytest.mark.parametrize("missing", [GATE_C, APP_C, BROWSER_C, DASH_C])
def test_check_fails_names_when_a_template_container_is_missing(stack: Host, missing: str):
    stack.update(lambda s: s["containers"].pop(missing))
    result = run_check(stack)
    assert result.returncode == 1
    assert "names-on-template" in failed_checks(result)
    assert list(verdicts(result)) == CHECKS


def test_check_does_not_pass_when_the_hermes_command_is_missing(stack: Host):
    (stack.root / "ctr/venv/hermes").unlink()
    result = run_check(stack)
    assert result.returncode == 1
    assert failed_checks(result) == {"names-on-template", "dispatcher-off", "cron-lists-empty",
                                     "no-messaging-platform", "telegram-send-fails"}


@pytest.mark.parametrize("code", [0, 2, 3, 126, 127])
def test_check_telegram_send_passes_only_when_delivery_fails(stack: Host, code: int):
    stack.update(lambda s: s["hermes"].update(send_rc=code))
    result = run_check(stack)
    assert result.returncode == 1
    assert failed_checks(result) == {"telegram-send-fails"}


def test_check_production_get_uses_one_assertion_that_stays_off_every_command_line(clean_check: CleanCheck):
    stack, before, result = clean_check
    assert result.returncode == 0, result.stdout + result.stderr
    minting =[c for c in stack.calls("docker") if docker_target(c) == ("exec", PROXY_C)]
    assert len(minting) == 1
    argv = minting[0]["argv"]
    assert "-i" in argv and "--user" not in argv
    after_name = argv[argv.index(PROXY_C) + 1:]
    assert after_name[:2] == ["python3", "-c"] and len(after_name) == 3
    assert "__main__" not in after_name[2]
    assert minting[0]["stdin"] == PROXY_SOURCE.read_bytes()
    jwt = minting[0]["stdout"].strip()
    assert re.fullmatch(r"[\w-]+\.[\w-]+\.[\w-]+", jwt)

    curl = stack.calls("curl")
    assert len(curl) == 1
    flags = curl[0]["argv"]
    assert flags[0] == "--disable", "curl reads a default configuration unless --disable is its first argument"
    assert flags[flags.index("-K") + 1] == "-"
    assert flags[flags.index("--max-redirs") + 1] == "0"
    assert flags[flags.index("-o") + 1] == "/dev/null"
    assert flags[flags.index("--max-time") + 1] == "20"
    assert flags[flags.index("-w") + 1] == "%{http_code} %{redirect_url}"
    assert not {"-L", "--location", "-H", "--header"} & set(flags)
    assert jwt.encode() in curl[0]["stdin"] and PROD_ORIGIN.encode() in curl[0]["stdin"]
    assert b"cf-access-jwt-assertion" in curl[0]["stdin"].lower()

    for secret in (jwt, PROD_ORIGIN, PROD_ORIGIN.split("//")[1]):
        assert secret not in argv_text(stack), "a value reached a command line"
        assert secret not in all_output(result), "a value was printed"
    assert jwt not in "".join(c.get("stderr", "") for c in stack.calls("docker"))
    assert snapshot(stack) == before, "the probe wrote to a volume"

    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding

    head, body, signature = jwt.split(".")
    key = serialization.load_pem_private_key(signing_key_pem(), password=None)
    key.public_key().verify(b64d(signature), f"{head}.{body}".encode(), padding.PKCS1v15(), hashes.SHA256())
    claims = json.loads(b64d(body))
    assert claims["iss"] == proxy_issuer() and claims["type"] == "app"


# The production probe with the real curl: the fake curl never reads a default curl configuration, so these
# tests send the probe to a throw-away HTTPS endpoint on 127.0.0.1 and hand curl a .curlrc through HOME.


class Seen(NamedTuple):
    method: str
    assertion: str | None
    body_bytes: int


class Refusing(http.server.BaseHTTPRequestHandler):
    """Answers every request, whatever its method, with 403 and no body, and records what arrived."""

    def __getattr__(self, name):
        if name.startswith("do_"):
            return self.refuse
        raise AttributeError(name)

    def refuse(self):
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        self.server.seen.append(Seen(self.command, self.headers.get("Cf-Access-Jwt-Assertion"), len(body)))
        self.send_response(403)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, *args):
        pass


class LocalHttps(NamedTuple):
    origin: str
    ca_bundle: Path
    seen: list[Seen]


def write_local_certificates(folder: Path) -> tuple[Path, Path, Path]:
    """A throw-away CA and a server certificate for 127.0.0.1 and localhost; returns the PEM files (ca, cert, key)."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    ca_key, server_key = ec.generate_private_key(ec.SECP256R1()), ec.generate_private_key(ec.SECP256R1())
    ca_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "journeys isolation test CA")])
    server_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")])
    now = datetime.datetime.now(datetime.timezone.utc)

    def issue(subject, key, *extensions):
        builder = (
            x509.CertificateBuilder().subject_name(subject).issuer_name(ca_name)
            .public_key(key.public_key()).serial_number(x509.random_serial_number())
            .not_valid_before(now - datetime.timedelta(hours=1)).not_valid_after(now + datetime.timedelta(days=1))
        )
        for extension, critical in extensions:
            builder = builder.add_extension(extension, critical)
        return builder.sign(ca_key, hashes.SHA256())

    ca_cert = issue(ca_name, ca_key, (x509.BasicConstraints(ca=True, path_length=None), True))
    server_cert = issue(
        server_name, server_key,
        (x509.BasicConstraints(ca=False, path_length=None), True),
        (x509.SubjectAlternativeName([x509.IPAddress(ipaddress.ip_address("127.0.0.1")), x509.DNSName("localhost")]),
         False),
    )
    folder.mkdir()
    pem = serialization.Encoding.PEM
    key_pem = server_key.private_bytes(pem, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
    files = folder / "ca.pem", folder / "server.pem", folder / "server-key.pem"
    for path, data in zip(files, (ca_cert.public_bytes(pem), server_cert.public_bytes(pem), key_pem)):
        path.write_bytes(data)
    return files


@pytest.fixture
def local_https(tmp_path: Path):
    """The endpoint in a thread of the test process, on an ephemeral port, trusted through its own CA."""
    ca, cert, key = write_local_certificates(tmp_path / "tls")
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert, key)
    server = http.server.HTTPServer(("127.0.0.1", 0), Refusing)
    server.socket = context.wrap_socket(server.socket, server_side=True)
    server.seen = []
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    try:
        yield LocalHttps(f"https://127.0.0.1:{server.server_address[1]}", ca, server.seen)
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


@pytest.mark.parametrize("curlrc", ["", 'request = "POST"\ndata = "write=1"\n'],
                         ids=["empty-default-config", "curlrc-requests-post"])
def test_check_production_get_with_the_real_curl_is_one_get_whatever_the_default_curl_configuration(
        stack: Host, local_https: LocalHttps, tmp_path: Path, curlrc: str):
    curl = shutil.which("curl")
    assert curl, "the real curl is not on PATH"
    stack.use_real_curl(curl)
    operator_home = tmp_path / "operator-home"
    operator_home.mkdir()
    (operator_home / ".curlrc").write_text(curlrc)

    # A None value removes the variable from the environment of the check: no ambient setting decides the result.
    env = dict.fromkeys(["SSL_CERT_FILE", "SSL_CERT_DIR", "CURL_HOME", "XDG_CONFIG_HOME", "HTTPS_PROXY",
                         "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"])
    env.update(HOME=str(operator_home), CURL_CA_BUNDLE=str(local_https.ca_bundle),
               NO_PROXY="127.0.0.1,localhost", JOURNEY_PRODUCTION_ORIGIN=local_https.origin)
    result = run_check(stack, **env)

    assert result.returncode == 0, result.stdout + result.stderr
    assert verdicts(result)["production-get-refused"] == "PASS"
    minting = [c for c in stack.calls("docker") if docker_target(c) == ("exec", PROXY_C)]
    assert len(minting) == 1
    jwt = minting[0]["stdout"].strip()
    assert re.fullmatch(r"[\w-]+\.[\w-]+\.[\w-]+", jwt)

    assert [(s.method, s.body_bytes) for s in local_https.seen] == [("GET", 0)]
    assert local_https.seen[0].assertion == jwt
    assert len(stack.calls("curl")) == 1, "the wrapper did not record the one curl call"
    assert jwt not in all_output(result), "the assertion was printed"
    assert jwt not in argv_text(stack), "the assertion reached a command line"


def test_check_never_creates_a_signing_key(stack: Host):
    (stack.vol(PROXY_V) / "signing-key.pem").unlink()
    before = snapshot(stack)
    result = run_check(stack)
    assert result.returncode == 1
    assert failed_checks(result) == {"production-get-refused"}
    assert snapshot(stack) == before
    assert stack.calls("curl") == []


@pytest.mark.parametrize("how", ["missing", "stopped"])
def test_check_production_get_fails_without_the_proxy_container(stack: Host, how: str):
    if how == "missing":
        stack.update(lambda s: s["containers"].pop(PROXY_C))
    else:
        stack.update(lambda s: s["containers"][PROXY_C].update(running=False))
    result = run_check(stack)
    assert result.returncode == 1
    assert "production-get-refused" in failed_checks(result)
    assert stack.calls("curl") == []


# ------------------------------------------------------------------ isolation_check.sh server1


NO_STACK_ENV = {"JOURNEY_STACK_HOST": None, "JOURNEY_PRODUCTION_ORIGIN": None}


def test_check_server1_passes_when_production_names_another_issuer(host: Host):
    before = snapshot(host)
    result = host.run(CHECK, "server1", env=NO_STACK_ENV)
    assert (result.returncode, result.stdout, result.stderr) == (0, "PASS production-issuer-differs\n", "")
    calls = host.calls()
    assert [c["tool"] for c in calls] == ["docker"]
    assert calls[0]["argv"] == ["exec", PROD_C, "printenv", "RAPHAEL_ACCESS_ISSUER"]
    for value in (PROD_ISSUER, proxy_issuer()):
        assert value not in argv_text(host) and value not in all_output(result)
    assert snapshot(host) == before


@pytest.mark.parametrize("case", ["same", "empty", "unset", "container-missing", "container-stopped"])
def test_check_server1_fails_closed(host: Host, case: str):
    def mutate(state: dict) -> None:
        box = state["containers"][PROD_C]
        if case == "same":
            box["env"] = [f"RAPHAEL_ACCESS_ISSUER={proxy_issuer()}"]
        elif case == "empty":
            box["env"] = ["RAPHAEL_ACCESS_ISSUER="]
        elif case == "unset":
            box["env"] = ["PATH=/usr/bin"]
        elif case == "container-missing":
            state["containers"].pop(PROD_C)
        else:
            box["running"] = False
    host.update(mutate)
    result = host.run(CHECK, "server1", env=NO_STACK_ENV)
    assert result.returncode == 1
    assert re.fullmatch(r"FAIL production-issuer-differs: \S.*\n", result.stdout)
    for value in (PROD_ISSUER, proxy_issuer()):
        assert value not in all_output(result) and value not in argv_text(host)


def test_check_server1_touches_no_stack_container_and_names_only_production(host: Host):
    host.run(CHECK, "server1", env=NO_STACK_ENV)
    named = {part for c in host.calls("docker") for part in c["argv"] if part.startswith("raphael-")}
    assert named == set(parse_template()["production-container"])


# ------------------------------------------------------------------ served profiles and the real cron list

EXTRA = "extra-one"
EXTRA_KEY = "extra-profile-key-0123456789abcdef"
JOB_PROGRAM = (
    "import sys\n"
    "from cron.jobs import create_job, pause_job, update_job\n"
    "job = create_job(prompt='p', schedule='every 1h', name='journey-test')\n"
    "if sys.argv[1] == 'disabled':\n    update_job(job['id'], {'enabled': False})\n"
    "elif sys.argv[1] == 'paused':\n    pause_job(job['id'])\n"
)


def add_served_profile(host: Host, name: str = EXTRA, dispatcher: bool = True, platform: bool = False) -> Path:
    folder = host.vol(HOME_V) / "profiles" / name
    folder.mkdir(parents=True)
    (folder / ".env").write_text(f"API_SERVER_KEY={EXTRA_KEY}\n")
    config: dict = {"kanban": {"dispatch_in_gateway": dispatcher}}
    if platform:
        config["platforms"] = {"discord": {"enabled": True}}
    (folder / "config.yaml").write_text(yaml.safe_dump(config))
    return folder


def fail_line(result: subprocess.CompletedProcess, check: str) -> str:
    return next(line for line in result.stdout.splitlines() if line.startswith(f"FAIL {check}:"))


@pytest.fixture(scope="module")
def real_jobs(tmp_path_factory) -> dict[str, Path]:
    # one real cron store for each kind of job, made once with the platform's own functions
    REAL_PYC.mkdir(parents=True, exist_ok=True)
    stores = {}
    for kind in ("enabled", "disabled", "paused"):
        home = tmp_path_factory.mktemp(f"real-cron-{kind}")
        env = {"PATH": os.environ["PATH"], "HOME": str(home), "HERMES_HOME": str(home), "PYTHONPATH": str(REPO),
               "PYTHONPYCACHEPREFIX": str(REAL_PYC), "AWS_EC2_METADATA_DISABLED": "true"}
        done = subprocess.run([sys.executable, "-c", JOB_PROGRAM, kind], env=env, cwd=home,
                              capture_output=True, text=True)
        assert done.returncode == 0, done.stdout + done.stderr
        assert (home / "cron").is_dir()
        stores[kind] = home / "cron"
    return stores


@pytest.fixture
def real_stack(stack: Host) -> Host:
    hermes_gives(real_platform=True)(stack)
    return stack


def test_check_passes_the_cron_check_on_the_real_empty_job_list(real_stack: Host):
    result = run_check(real_stack)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.splitlines() == [f"PASS {name}" for name in CHECKS]


@pytest.mark.parametrize("kind, profile", [("disabled", "default"), ("disabled", "planner"),
                                           ("disabled", "designer"), ("paused", "default")])
def test_check_fails_on_any_real_cron_job_even_a_disabled_or_paused_one(real_stack: Host, real_jobs, kind, profile):
    # the reset's own `hermes project create` already made the default home's cron folder (empty); the job goes into it
    shutil.copytree(real_jobs[kind], profile_folder(real_stack, profile) / "cron", dirs_exist_ok=True)
    result = run_check(real_stack)
    name = "default" if profile == "default" else f"raphael-{profile}"
    assert result.returncode == 1, result.stdout
    assert list(verdicts(result)) == CHECKS
    assert failed_checks(result) == {"cron-lists-empty"}
    assert f"FAIL cron-lists-empty: profile {name} has a cron job" in result.stdout.splitlines()


def test_check_lists_cron_jobs_with_the_all_option(clean_check: CleanCheck):
    listed = [c for c in clean_check.host.calls("docker")
              if c["argv"][0] == "exec" and any("hermes cron list" in part for part in c["argv"])]
    assert len(listed) == 1 + len(list((clean_check.host.vol(HOME_V) / "profiles").iterdir()))
    assert all(any(part.endswith("hermes cron list --all") for part in c["argv"]) for c in listed)


def test_check_fails_each_check_of_an_extra_served_profile_by_name(stack: Host):
    add_served_profile(stack)
    cron_job(EXTRA)(stack)
    before = snapshot(stack)
    result = run_check(stack)
    assert result.returncode == 1, result.stdout
    assert list(verdicts(result)) == CHECKS
    assert failed_checks(result) == {"names-on-template", "dispatcher-off", "cron-lists-empty"}
    for check in ("names-on-template", "dispatcher-off", "cron-lists-empty"):
        assert EXTRA in fail_line(result, check)
    assert snapshot(stack) == before
    assert EXTRA_KEY not in all_output(result)


def test_check_fails_the_messaging_platform_of_an_extra_served_profile_by_name(stack: Host):
    add_served_profile(stack, dispatcher=False, platform=True)
    result = run_check(stack)
    assert result.returncode == 1, result.stdout
    assert failed_checks(result) == {"names-on-template", "no-messaging-platform"}
    line = fail_line(result, "no-messaging-platform")
    assert EXTRA in line and "discord" in line


def test_check_tries_the_send_from_an_extra_served_profile_too(stack: Host):
    add_served_profile(stack, dispatcher=False)
    hermes_gives(send_rc=0)(stack)
    result = run_check(stack)
    assert failed_checks(result) == {"names-on-template", "telegram-send-fails"}
    assert EXTRA in fail_line(result, "telegram-send-fails")


def test_check_names_an_extra_served_profile_that_has_nothing_else_wrong(stack: Host):
    add_served_profile(stack, dispatcher=False)
    result = run_check(stack)
    assert result.returncode == 1
    assert failed_checks(result) == {"names-on-template"}
    assert EXTRA in fail_line(result, "names-on-template")


@pytest.mark.parametrize("entry, kind", [
    ("Upper-Case", "dir"), ("a" * 70, "dir"), ("notes.txt", "file"), (".hidden", "dir"),
    ("bad$(touch PWNED)", "dir"),
])
def test_check_fails_closed_for_any_entry_of_the_profiles_folder_that_is_not_a_permitted_profile(stack: Host, entry, kind):
    target = stack.vol(HOME_V) / "profiles" / entry
    target.mkdir() if kind == "dir" else target.write_text("not a profile\n")
    result = run_check(stack)
    assert result.returncode == 1, result.stdout
    assert list(verdicts(result)) == CHECKS
    assert failed_checks(result) == {"names-on-template"}
    assert not list(stack.root.rglob("PWNED")), "a profile name reached a shell"
    line = fail_line(result, "names-on-template")
    if re.fullmatch(r"[A-Za-z0-9._-]+", entry):
        assert entry in line
    else:
        assert "bad;touch" not in all_output(result) and "bad$(touch" not in all_output(result)
        assert "bad;touch" not in argv_text(stack) and "bad$(touch" not in argv_text(stack)


def test_check_reports_a_served_name_that_is_not_plain_and_never_passes_it_to_a_shell(stack: Host):
    hermes_gives(serve_extra=["Bad;touch PWNED"])(stack)
    result = run_check(stack)
    assert result.returncode == 1, result.stdout
    assert list(verdicts(result)) == CHECKS
    assert failed_checks(result) == {"names-on-template"}
    assert not list(stack.root.rglob("PWNED"))
    assert "Bad;touch" not in all_output(result) and "Bad;touch" not in argv_text(stack)


def test_check_fails_names_when_the_served_profiles_cannot_be_read(stack: Host):
    hermes_gives(serve_fail=True)(stack)
    result = run_check(stack)
    assert result.returncode == 1, result.stdout
    assert list(verdicts(result)) == CHECKS
    assert failed_checks(result) == {"names-on-template"}
    assert "the served profiles cannot be read" in fail_line(result, "names-on-template")


def test_check_still_checks_both_permitted_profiles_when_the_served_profiles_cannot_be_read(stack: Host):
    hermes_gives(serve_fail=True)(stack)
    dispatcher_on("planner")(stack)
    cron_job("default")(stack)
    result = run_check(stack)
    assert failed_checks(result) == {"names-on-template", "dispatcher-off", "cron-lists-empty"}
    assert "raphael-planner" in fail_line(result, "dispatcher-off")
    assert "default" in fail_line(result, "cron-lists-empty")


def test_check_passes_the_names_check_when_only_the_permitted_profiles_exist(clean_check: CleanCheck):
    assert sorted(p.name for p in (clean_check.host.vol(HOME_V) / "profiles").iterdir()) == [
        "raphael-designer", "raphael-planner"]
    assert verdicts(clean_check.result)["names-on-template"] == "PASS"


def cron_lists(host: Host) -> list[dict]:
    return [c for c in host.calls("docker")
            if c["argv"][0] == "exec" and any("hermes cron list" in part for part in c["argv"])]


def test_check_permits_the_designer_and_checks_it_like_the_planner(clean_check: CleanCheck):
    assert (clean_check.host.vol(HOME_V) / "profiles/raphael-designer").is_dir()
    assert clean_check.result.returncode == 0, clean_check.result.stdout
    assert verdicts(clean_check.result) == {name: "PASS" for name in CHECKS}
    # One cron list for each of default, planner and designer: the designer is read like the planner.
    assert len(cron_lists(clean_check.host)) == 3


def test_check_still_passes_on_a_stack_without_the_designer(stack: Host):
    shutil.rmtree(stack.vol(HOME_V) / "profiles/raphael-designer")
    assert sorted(p.name for p in (stack.vol(HOME_V) / "profiles").iterdir()) == ["raphael-planner"]
    result = run_check(stack)
    assert result.returncode == 0, result.stdout + result.stderr
    assert verdicts(result) == {name: "PASS" for name in CHECKS}
    # Only default and planner were read: the designer is checked when the stack has it, not required.
    assert len(cron_lists(stack)) == 2


@pytest.mark.parametrize("breaking", [dispatcher_on("designer"), dispatcher_unset("designer"),
                                      cron_job("designer"), platform_enabled("designer")])
def test_check_fails_the_designer_by_name_when_it_breaks_what_the_planner_may_not(stack: Host, breaking):
    breaking(stack)
    result = run_check(stack)
    assert result.returncode == 1, result.stdout
    (failed,) = failed_checks(result)
    assert "raphael-designer" in fail_line(result, failed)


def test_check_still_checks_the_designer_when_the_served_profiles_cannot_be_read(stack: Host):
    hermes_gives(serve_fail=True)(stack)
    dispatcher_on("designer")(stack)
    result = run_check(stack)
    assert failed_checks(result) == {"names-on-template", "dispatcher-off"}
    assert "raphael-designer" in fail_line(result, "dispatcher-off")


def copy_scripts(dest: Path, rename: dict[str, str] | None = None) -> Path:
    """A copy of the scripts folder (scripts, template, stack-home, proxy), names optionally replaced."""
    target = dest / "scripts" / "journeys"
    target.mkdir(parents=True)
    for name in ("reset_stack.sh", "isolation_check.sh", "stack.env.template", "synthetic_proxy.py"):
        shutil.copy2(JOURNEYS / name, target / name)
    shutil.copytree(STACK_HOME, target / "stack-home")
    for old, new in (rename or {}).items():
        template = target / "stack.env.template"
        template.write_text(template.read_text().replace(old, new))
    return target


@pytest.mark.parametrize("edit", ["no-issuer-line", "empty-issuer"])
def test_check_server1_fails_when_the_proxy_issuer_cannot_be_read(host: Host, tmp_path: Path, edit: str):
    copy = copy_scripts(tmp_path / "copy")
    proxy = copy / "synthetic_proxy.py"
    text = proxy.read_text()
    if edit == "no-issuer-line":
        text = re.sub(r"^ISSUER = .*$", "ISSUER_RENAMED = 1", text, flags=re.M)
    else:
        text = re.sub(r'^ISSUER = ".*"$', 'ISSUER = ""', text, flags=re.M)
    proxy.write_text(text)
    result = host.run(copy / "isolation_check.sh", "server1", env=NO_STACK_ENV)
    assert result.returncode == 1
    assert result.stdout.startswith("FAIL production-issuer-differs")


# ------------------------------------------------------------------ names come only from the template


def test_both_scripts_take_every_name_from_the_template(tmp_path: Path):
    alt = Host(tmp_path / "host")
    renames = {"raphael-journeys-": "alt-rig-", PROD_C: "alt-production"}
    text = (alt.root / "state.json").read_text()
    for old, new in renames.items():
        text = text.replace(old, new)
    (alt.root / "state.json").write_text(text)
    (alt.root / "owners.json").write_text((alt.root / "owners.json").read_text().replace("raphael-journeys-", "alt-rig-"))
    write_view(json.loads(text), alt.root)
    for folder in sorted((alt.root / "volumes").iterdir()):
        if folder.name.startswith("raphael-journeys-"):
            folder.rename(folder.with_name(folder.name.replace("raphael-journeys-", "alt-rig-")))
    copy = copy_scripts(tmp_path / "copy", renames)

    reset = alt.run(copy / "reset_stack.sh", env={"JOURNEY_MODEL_TOKEN_FILE": str(alt.model_token_file())})
    assert reset.returncode == 0, reset.stdout + reset.stderr
    server2 = alt.run(copy / "isolation_check.sh", "server2")
    server1 = alt.run(copy / "isolation_check.sh", "server1")
    assert server2.returncode == 0, server2.stdout + server2.stderr
    assert server2.stdout.splitlines() == [f"PASS {name}" for name in CHECKS]
    assert server1.returncode == 0, server1.stdout + server1.stderr

    parts = [part for c in alt.calls("docker") for part in c["argv"]]
    named = set(re.findall(r"alt-rig-[a-z]+|alt-production", "\n".join(parts)))
    assert {"alt-rig-home", "alt-rig-tokens", "alt-rig-proxy", "alt-production"} <= named
    assert not set().union(*(docker_names(c) for c in alt.calls("docker"))), "an old name was used"
    assert (alt.vol("alt-rig-home") / "config.yaml").is_file()
    assert (alt.vol("alt-rig-tokens") / "workspace").is_file()
