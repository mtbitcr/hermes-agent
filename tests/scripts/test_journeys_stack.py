"""Tests for the journeys stack reset and isolation scripts (card P-2).

The scripts run against a fake docker, hostname, curl and sleep that sit first on PATH in a temp
folder. The "host" is a folder tree: containers and volumes are entries in a state file, volumes
are plain folders, and the shell snippets the scripts send into containers really run locally on
those folders. No docker, no network and no real Server 2 is involved.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest
import yaml

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
# Old rig names (facts file): present on the fake host so that any call naming them is caught.
DECOY_CONTAINERS = [
    "raphael-s2-workspace", "raphael-s2-proxy", "raphael-s2-dashboard",
    "raphael-s2-harness", "raphael-s2-browser",
]
DECOY_VOLUMES = ["raphael-s2-state", "raphael-s2-state-prod", "raphael-s2-pw"]
# Profile folder names are platform data, not container or volume names.
NOT_RESOURCE_NAMES = {"raphael-planner"}

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


FAKE_COMMON = r'''
import json, os, re, subprocess, sys, time
from pathlib import Path

ROOT = Path(os.environ["FAKE_HOST_ROOT"])
STATE = ROOT / "state.json"
LOG = ROOT / "calls.jsonl"


def load():
    return json.loads(STATE.read_text())


def save(state):
    tmp = STATE.with_name("state.json.tmp")
    tmp.write_text(json.dumps(state))
    os.replace(tmp, STATE)


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
    if sub == "restart" and entry.get("health"):
        entry["starting_polls"] = entry.get("health_delay", 0)
    save(state)
    return finish(seq, 0, (args[0] + "\n").encode())


def unmap(data, pairs):
    """Show the container's view: host folders in the output become the mount destinations."""
    text = data.decode("utf-8", "replace")
    for dest, host in pairs:
        text = text.replace(host, dest)
    return text.encode()


def container_env(entry, pairs):
    env = {"PATH": "%s:%s:%s:/usr/bin:/bin" % (ROOT / "ctr" / "shim", ROOT / "ctr" / "venv",
                                               ROOT / "ctr" / "bin"),
           "FAKE_HOST_ROOT": str(ROOT), "PYTHONPATH": str(ROOT / "ctr" / "pkgs"),
           "PYTHONDONTWRITEBYTECODE": "1", "HOME": str(ROOT / "ctr" / "home")}
    for item in entry["env"]:
        key, _, value = item.partition("=")
        if key != "PATH":
            env[key] = map_text(value, pairs)
    return env


def d_exec(argv, args):
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
    pairs = mount_pairs(entry)
    local = [map_text(part, pairs) for part in cmd]
    if local[0] in ("python3", "python"):
        local[0] = sys.executable
    try:
        done = subprocess.run(local, input=stdin_bytes or b"", capture_output=True,
                              env=container_env(entry, pairs), cwd=str(ROOT / "ctr" / "home"))
    except FileNotFoundError:
        return finish(seq, 127, err=b"exec: command not found\n")
    return finish(seq, done.returncode, unmap(done.stdout, pairs), unmap(done.stderr, pairs))


def d_run(argv, args):
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


def issue_token(args):
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
    if load().get("hermes", {}).get("issue_fail") == surface:
        sys.stderr.write("fake hermes: issue failed\n")
        return 1
    parent = Path(out).parent
    if not parent.is_dir() or stat.S_IMODE(parent.stat().st_mode) & 0o077:
        sys.stderr.write("fake hermes: parent folder must exist and be owner-only\n")
        return 1
    fd = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "w") as handle:
        handle.write("tok-%s-%s" % (surface, os.urandom(12).hex()))
    os.chmod(out, 0o600)
    print("issued surface=%s" % surface)
    return 0


def hermes_main(args):
    spec = load().get("hermes", {})
    home = Path(os.environ["HERMES_HOME"])
    if args[:2] == ["cron", "list"]:
        profile = "planner" if home.name == "raphael-planner" else "default"
        jobs = spec.get("cron_jobs", {}).get(profile, [])
        sys.stdout.write(EMPTY_CRON if not jobs else "".join("%s\n" % job for job in jobs))
        return spec.get("cron_rc", 0)
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
             "curl": tool_curl}
    sys.exit(tools[sys.argv[1]](sys.argv[2:]))


if __name__ == "__main__":
    main()
'''

FAKE_TOOLS_SOURCE = (
    FAKE_COMMON
    + "".join("%s = %r\n" % (n, globals()[n])
              for n in ("F_IMAGE", "F_ENV_NAMES", "F_MOUNT_TYPES", "F_MOUNT_DEST", "F_STATE"))
    + FAKE_DOCKER
    + FAKE_PLATFORM
)

# Stand-ins for the platform modules the scripts import inside the container.
FAKE_PACKAGES = {
    "hermes_cli/__init__.py": "",
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


def docker_target(call: dict) -> tuple[str, str | None]:
    """Subcommand and container name of a recorded docker call (None when it names none)."""
    argv = call["argv"]
    sub = argv[0]
    if sub in ("start", "restart", "stop"):
        return sub, argv[1]
    if sub == "inspect":
        return sub, argv[-1]
    if sub == "exec":
        i = 1
        while argv[i].startswith("-"):
            i += 2 if argv[i] in ("-u", "--user") else 1
        return sub, argv[i]
    return sub, None


class Host:
    """A fake Server 2: state file, volume folders, fake tools and a way to run the scripts."""

    def __init__(self, root: Path, build: bool = True):
        self.root = root
        if build:
            self._build()

    @classmethod
    def copy_of(cls, other: "Host", root: Path) -> "Host":
        """A second host with the same state and volumes, and an empty call log."""
        shutil.copytree(other.root, root, symlinks=True)
        clone = cls(root, build=False)
        clone._write_tools()
        clone.clear_calls()
        (root / "interp.log").unlink(missing_ok=True)
        return clone

    @staticmethod
    def _exe(path: Path, text: str) -> None:
        path.write_text(text)
        path.chmod(0o755)

    def _write_tools(self) -> None:
        root = self.root
        (root / "fake_tools.py").write_text(FAKE_TOOLS_SOURCE)
        for tool in ("docker", "hostname", "curl", "sleep"):
            self._exe(root / "bin" / tool,
                      f'#!/bin/sh\nexec "{sys.executable}" "$FAKE_HOST_ROOT/fake_tools.py" {tool} "$@"\n')
        # Like the published image: a /bin/sh shim sits first on PATH and runs the venv binary,
        # whose shebang names the venv's own interpreter (a wrapper here, so the tests see its use).
        self._exe(root / "ctr/shim/hermes", '#!/bin/sh\nexec "$FAKE_HOST_ROOT/ctr/venv/hermes" "$@"\n')
        self._exe(root / "ctr/venv/python",
                  f'#!/bin/sh\nprintf \'%s\\n\' "$1" >> "$FAKE_HOST_ROOT/interp.log"\nexec "{sys.executable}" "$@"\n')
        self._exe(root / "ctr/venv/hermes",
                  f"#!{root}/ctr/venv/python\nimport os, sys\nsys.path.insert(0, os.environ['FAKE_HOST_ROOT'])\n"
                  "import fake_tools\nsys.exit(fake_tools.hermes_main(sys.argv[1:]))\n")
        self._exe(root / "ctr/bin/chown",
                  '#!/bin/sh\nprintf \'%s\\n\' "$*" >> "$FAKE_HOST_ROOT/chown.log"\nexit 0\n')

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
        (root / "state.json").write_text(json.dumps(self._initial_state()))

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

        def box(image, env, mounts, **extra):
            return {"image": image, "env": img[image] + env, "mounts": mounts, "running": True, **extra}

        containers = {
            PROXY_C: box("sha256:img-proxy", ["SYNTHETIC_STATE=/state"], [vol(PROXY_V, "/state")]),
            APP_C: box("sha256:img-app", app_env, [vol(TOKENS_V, "/run/tokens")]),
            DASH_C: box("sha256:img-hermes", ["HERMES_HOME=/opt/data"],
                        [vol(HOME_V, "/opt/data"), vol(TOKENS_V, "/tokens")],
                        health="healthy", health_delay=1),
            GATE_C: box("sha256:img-hermes", ["HERMES_HOME=/opt/data"], [vol(HOME_V, "/opt/data")]),
            BROWSER_C: box("sha256:img-browser",
                           ["JOURNEY_BASE_URL=https://workspace.synthetic.test",
                            f"JOURNEY_PRODUCTION_ORIGIN={PROD_ORIGIN}", "PLAYWRIGHT_BROWSERS_PATH=/pw"], []),
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

    # --- reading and changing the fake host
    def state(self) -> dict:
        return json.loads((self.root / "state.json").read_text())

    def update(self, mutate) -> None:
        state = self.state()
        mutate(state)
        (self.root / "state.json").write_text(json.dumps(state))

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


def test_stack_home_sets_what_the_facts_say():
    default = yaml.safe_load((STACK_HOME / "config.yaml").read_text())
    profiles = sorted(p for p in (STACK_HOME / "profiles").iterdir() if p.is_dir())
    assert [p.name for p in profiles] == ["raphael-planner"]
    planner = yaml.safe_load((profiles[0] / "config.yaml").read_text())
    multiplex = default.get("multiplex_profiles", (default.get("gateway") or {}).get("multiplex_profiles"))
    assert multiplex is True
    for config in (default, planner):
        assert config["kanban"]["dispatch_in_gateway"] is False
        platforms = config.get("platforms") or {}
        assert set(platforms) <= {"api_server"}, "no messaging platform may be configured"
        assert "cron" not in config


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
        assert sub in ("inspect", "exec"), f"changing docker call: {call['argv'][:3]}"
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


def test_reset_refuses_when_hermes_home_is_not_the_home_mount(host: Host):
    def mutate(state: dict) -> None:
        box = state["containers"][DASH_C]
        box["env"] = ["HERMES_HOME=/somewhere/else" if e.startswith("HERMES_HOME=") else e for e in box["env"]]
    host.update(mutate)
    before = snapshot(host)
    result = run_reset(host)
    assert result.returncode != 0
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


def test_reset_makes_a_fresh_home_and_token_folder(host: Host):
    untouched = {v: sha_tree(host.vol(v)) for v in [PROXY_V] + DECOY_VOLUMES}
    result = reset_ok(host)
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

    chown = (host.root / "chown.log").read_text().splitlines()
    assert f"-R 1000:1000 {home}" in chown and f"1000:1000 {tokens}" in chown
    assert {v: sha_tree(host.vol(v)) for v in untouched} == untouched
    assert not host.state().get("created_volumes")
    assert_secret_free(host, result, generated_secrets(host))


def test_reset_can_run_twice_and_makes_new_values(host: Host):
    reset_ok(host)
    first = generated_secrets(host)
    result = reset_ok(host)
    second = generated_secrets(host)
    assert len(second) == len(first)
    assert not (set(first) - {MODEL_TOKEN}) & set(second)
    for path in home_files(host).values():
        assert parse_env_file(path)[0].count("API_SERVER_KEY") == 1
    assert_secret_free(host, result, first + second)


def test_reset_order_of_stops_writes_and_starts(host: Host):
    reset_ok(host)
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
    assert steps[5:-5] == [("write", DASH_C)] * 9  # 5 tokens, 2 API server keys, 2 model tokens
    assert steps[-5:-2] == [("restart", APP_C), ("restart", DASH_C), ("start", GATE_C)]
    assert set(steps[-2:]) == {("start", PROXY_C), ("start", BROWSER_C)}


def test_reset_helpers_have_the_agreed_shape(host: Host):
    reset_ok(host)
    image = host.state()["containers"][DASH_C]["image"]
    runs = [c for c in host.calls("docker") if c["argv"][0] == "run"]
    assert sorted(mount_source(c) for c in runs) == sorted([HOME_V, TOKENS_V])
    for call in runs:
        argv, source = call["argv"], mount_source(call)
        flags = ["--rm", "-i"] if source == HOME_V else ["--rm"]
        assert argv[:-1] == ["run", *flags, "--network", "none", "--user", "0", "--entrypoint", "sh",
                             "--mount", f"type=volume,source={source},target=/volume", image, "-c"]
        script = argv[-1]
        assert "find /volume -mindepth 1 -delete" in script
        if source == HOME_V:
            assert "chown -R 1000:1000 /volume" in script and "chmod go-rwx /volume" in script
            with tarfile.open(fileobj=io.BytesIO(call["stdin"])) as archive:
                members = {m.name.removeprefix("./") for m in archive.getmembers() if m.isfile()}
            assert members == {p.relative_to(STACK_HOME).as_posix() for p in STACK_HOME.rglob("*")
                               if p.is_file()}
        else:
            assert "chown 1000:1000 /volume" in script and "chmod 700 /volume" in script
            assert not call["stdin"]


def test_reset_writers_run_as_uid_1000_with_values_only_on_stdin(host: Host):
    reset_ok(host)
    writers = [c for c in host.calls("docker") if docker_target(c)[0] == "exec" and "--user" in c["argv"]]
    assert len(writers) == 9
    for call in writers:
        argv = call["argv"]
        assert argv[argv.index("--user") + 1] == "1000"
        assert docker_target(call)[1] == DASH_C
    issues = [c for c in writers if "kanban-workspace-token" in c["argv"]]
    assert sorted(c["argv"][c["argv"].index("--surface") + 1] for c in issues) == sorted(SURFACE_ENV)
    for call in issues:
        argv = call["argv"]
        assert not [a for a in argv if a.startswith("--ttl")]
        assert argv[argv.index("--out") + 1] == "/tokens/" + argv[argv.index("--surface") + 1]
    carrying = [c for c in writers if c["stdin"] and MODEL_TOKEN.encode() in c["stdin"]]
    assert len(carrying) == 2 and all("-i" in c["argv"] for c in carrying)


def test_reset_names_only_template_containers_and_volumes(host: Host):
    reset_ok(host)
    template = parse_template()
    allowed = set(template["containers"]) | set(template["volumes"])
    calls = host.calls("docker")
    found = set().union(*(resource_names_in(part) for c in calls for part in c["argv"]))
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
    for script in (RESET, CHECK):
        for number, line in enumerate(script.read_text().splitlines(), 1):
            if not line.lstrip().startswith("#"):
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
    assert len(writer_calls(host)) == 7  # 5 tokens and 2 API server keys; no model token writes


def test_reset_stops_when_a_token_cannot_be_issued(host: Host):
    host.update(lambda state: state["hermes"].update(issue_fail="connections"))
    result = run_reset(host)
    assert result.returncode != 0
    assert "connections" in all_output(result)
    done = [docker_target(c) for c in host.calls("docker")]
    assert not [step for step in done if step[0] == "restart"]
    assert ("start", GATE_C) not in done
    assert_secret_free(host, result, [MODEL_TOKEN])


def test_reset_prints_no_skip_line_when_the_model_token_is_written(host: Host):
    result = reset_ok(host)
    assert "skip" not in all_output(result).lower()
    assert len(writer_calls(host)) == 9


def test_reset_leaves_the_repository_files_alone(host: Host):
    before = sha_tree(STACK_HOME)
    reset_ok(host)
    assert sha_tree(STACK_HOME) == before


# ------------------------------------------------------------------ isolation_check.sh


@pytest.fixture(scope="module")
def reset_tree(tmp_path_factory) -> Host:
    """One reset stack, made once; each test below works on its own copy of it."""
    first = Host(tmp_path_factory.mktemp("after-reset") / "host")
    result = first.run(RESET, env={"JOURNEY_MODEL_TOKEN_FILE": str(first.model_token_file())})
    assert result.returncode == 0, result.stdout + result.stderr
    return first


@pytest.fixture
def stack(reset_tree: Host, tmp_path: Path) -> Host:
    return Host.copy_of(reset_tree, tmp_path / "host")


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
    return home if profile == "default" else home / "profiles" / "raphael-planner"


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
    "home-env-line-without-name": ("names-on-template", lambda h: add_env_line(h, "planner", "just-text"), {}),
    "dispatcher-on-default": ("dispatcher-off", dispatcher_on("default"), {}),
    "dispatcher-on-planner": ("dispatcher-off", dispatcher_on("planner"), {}),
    "dispatcher-unset-planner": ("dispatcher-off", dispatcher_unset("planner"), {}),
    "cron-job-default": ("cron-lists-empty", cron_job("default"), {}),
    "cron-job-planner": ("cron-lists-empty", cron_job("planner"), {}),
    "cron-command-fails": ("cron-lists-empty", hermes_gives(cron_rc=1), {}),
    "platform-in-default-config": ("no-messaging-platform", platform_enabled("default"), {}),
    "platform-in-planner-config": ("no-messaging-platform", platform_enabled("planner"), {}),
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


def test_check_server2_passes_every_check_on_a_clean_stack(stack: Host):
    before = snapshot(stack)
    result = run_check(stack)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.splitlines() == [f"PASS {name}" for name in CHECKS]
    assert result.stderr == ""
    assert snapshot(stack) == before


def test_check_server2_names_only_template_entries(stack: Host):
    result = run_check(stack)
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


def test_check_reads_the_config_with_the_interpreter_of_the_hermes_command(stack: Host):
    run_check(stack)
    uses = (stack.root / "interp.log").read_text().splitlines()
    assert uses.count("-") >= 4, "the dispatcher and platform reads, for both profiles, use that interpreter"


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
    assert failed_checks(result) == {"dispatcher-off", "cron-lists-empty", "no-messaging-platform",
                                     "telegram-send-fails"}


@pytest.mark.parametrize("code", [0, 2, 3, 126, 127])
def test_check_telegram_send_passes_only_when_delivery_fails(stack: Host, code: int):
    stack.update(lambda s: s["hermes"].update(send_rc=code))
    result = run_check(stack)
    assert result.returncode == 1
    assert failed_checks(result) == {"telegram-send-fails"}


def test_check_production_get_uses_one_assertion_that_stays_off_every_command_line(stack: Host):
    before = snapshot(stack)
    result = run_check(stack)
    assert result.returncode == 0, result.stdout + result.stderr
    minting = [c for c in stack.calls("docker") if docker_target(c) == ("exec", PROXY_C)]
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
    assert not set().union(*(resource_names_in(part) for part in parts)), "an old name was used"
    assert (alt.vol("alt-rig-home") / "config.yaml").is_file()
    assert (alt.vol("alt-rig-tokens") / "workspace").is_file()
