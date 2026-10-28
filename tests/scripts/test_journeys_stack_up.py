"""Tests for scripts/journeys/stack_up.sh (card 2 of the journeys stack plan).

The script runs against stand-ins for docker, sleep, hostname and df that sit first on PATH. The stand-in
docker records every call in calls.jsonl, keeps a fake Server 2 in state.json, and refuses (status 98) any
subcommand or flag the script is not expected to use, including the combinations real docker refuses with
--network container:. DOCKER_HOST points at a path that does not exist, so a real docker could not answer.
No real Docker call is made anywhere. The tests read recorded calls, output lines and the fake host state;
they never read the script's text.

Tests test_1_... to test_8_... are the eight tests of the plan; test_extra_... are additions.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.linux_only

REPO = Path(__file__).resolve().parents[2]
JOURNEYS = REPO / "scripts" / "journeys"
TEMPLATE = JOURNEYS / "stack.env.template"

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
NETWORK = "raphael-journeys-net"
LABEL = "raphael-journeys.made-by=stack_up"

HOST = "journeys-host.invalid"
PLATFORM = "0123456789abcdef0123456789abcdef01234567"
REVISION = "89abcdef0123456789abcdef0123456789abcdef"
NEW_REVISION = "fedcba98765432100123456789abcdef01234567"
NEW_PLATFORM = "76543210fedcba9876543210fedcba9876543210"

# The only docker --format shapes the scripts use. The stand-in answers exactly these.
FORMATS = {
    "image": "{{.Image}}",
    "env_names": '{{range .Config.Env}}{{println (index (split . "=") 0)}}{{end}}',
    "mount_types": "{{range .Mounts}}{{println .Type .Name}}{{end}}",
    "mount_dest": "{{range .Mounts}}{{println .Name .Destination}}{{end}}",
    "state": "{{.State.Running}} {{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}",
    "hermes_home": (
        '{{range .Config.Env}}{{$p := split . "="}}{{if eq (index $p 0) "HERMES_HOME"}}{{index $p 1}}{{end}}{{end}}'
    ),
    "health_kind": "{{if .Config.Healthcheck}}{{index .Config.Healthcheck.Test 0}}{{else}}NONE{{end}}",
    "ident": '{{.Id}} {{index .Config.Labels "raphael-journeys.made-by"}}',
    "config_image": "{{.Config.Image}}",
    "network_mode": "{{.HostConfig.NetworkMode}}",
    "name": "{{.Name}}",
    "id": "{{.Id}}",
    "security": "{{range .SecurityOptions}}{{println .}}{{end}}",
    "root_dir": "{{.DockerRootDir}}",
}

STUB = r'''
import hashlib
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATE_FILE = ROOT / "state.json"
CALLS = ROOT / "calls.jsonl"
LABEL_KEY = "raphael-journeys.made-by"
HEX64 = re.compile(r"[0-9a-f]{64}")
TOOL = ""


class Done(Exception):
    def __init__(self, rc, err=""):
        self.rc, self.err = rc, err


def refuse(why):
    raise Done(98, "stand-in %s: refused: %s\n" % (TOOL, why))


def fail(message, rc=1):
    raise Done(rc, message + "\n")


def load():
    return json.loads(STATE_FILE.read_text(encoding="utf-8"))


def save(state):
    tmp = STATE_FILE.with_name("state.tmp.%d" % os.getpid())
    tmp.write_text(json.dumps(state), encoding="utf-8")
    os.replace(tmp, STATE_FILE)


def log(argv, rc):
    line = json.dumps({"tool": TOOL, "argv": argv, "rc": rc}) + "\n"
    fd = os.open(CALLS, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        os.write(fd, line.encode())
    finally:
        os.close(fd)


def find_container(state, ref):
    for cid, c in state["containers"].items():
        if ref == cid or ref == c["name"]:
            return cid, c
    return None, None


def find_image(state, ref):
    for image_ref, img in state["images"].items():
        if ref == image_ref or ref == img["id"]:
            return image_ref, img
    return None, None


def merged_env(state, c):
    env = dict(find_image(state, c["image_id"])[1]["env"])
    env.update(c["env"])
    return env


def new_id(state, kind):
    state["seq"] += 1
    return hashlib.sha256(("%s-%d" % (kind, state["seq"])).encode()).hexdigest()


def join_check(state, c):
    if c["network"].startswith("container:"):
        target = c["network"][len("container:"):]
        _, t = find_container(state, target)
        if t is None or not t["running"]:
            fail("Error response from daemon: cannot join network of a non running container: " + target)


def d_info(state, a):
    if len(a) != 2 or a[0] != "--format":
        refuse("info shape %r" % a)
    if a[1] == F["security"]:
        return "".join(x + "\n" for x in state["security"]) + "\n"
    if a[1] == F["root_dir"]:
        return state["docker_root"] + "\n"
    refuse("info format")


def d_inspect(state, a):
    if len(a) != 5 or a[0] != "--type" or a[2] != "--format":
        refuse("inspect shape %r" % a)
    kind, fmt, ref = a[1], a[3], a[4]
    if kind == "container":
        cid, c = find_container(state, ref)
        if c is None:
            fail("Error: No such container: " + ref)
        env = merged_env(state, c)
        if fmt == F["image"]:
            out = c["image_id"]
        elif fmt == F["env_names"]:
            out = "".join(n + "\n" for n in env)
        elif fmt == F["mount_types"]:
            out = "".join("volume %s\n" % m["name"] for m in c["mounts"])
        elif fmt == F["mount_dest"]:
            out = "".join("%s %s\n" % (m["name"], m["dest"]) for m in c["mounts"])
        elif fmt == F["hermes_home"]:
            out = env.get("HERMES_HOME", "")
        elif fmt == F["health_kind"]:
            out = c["health"][0] if c["health"] else "NONE"
        elif fmt == F["ident"]:
            out = "%s %s" % (cid, c["labels"].get(LABEL_KEY, ""))
        elif fmt == F["config_image"]:
            out = c["image_ref"]
        elif fmt == F["network_mode"]:
            out = c["network"]
        elif fmt == F["state"]:
            if c["running"] and c["health"] and c["status"] == "starting":
                c["polls"] -= 1
                if c["polls"] <= 0 and c["name"] not in state["never_healthy"]:
                    c["status"] = "healthy"
                save(state)
            out = "%s %s" % ("true" if c["running"] else "false", c["status"] if c["health"] else "none")
        else:
            refuse("container format %r" % fmt)
        return out + "\n"
    if kind == "image":
        _, img = find_image(state, ref)
        if img is None:
            fail("Error: No such image: " + ref)
        if fmt == F["env_names"]:
            return "".join(n + "\n" for n in img["env"]) + "\n"
        if fmt == F["id"]:
            return img["id"] + "\n"
        refuse("image format %r" % fmt)
    if kind in ("network", "volume"):
        table = state["networks"] if kind == "network" else state["volumes"]
        if ref not in table:
            fail("Error: No such %s: %s" % (kind, ref))
        if fmt != F["name"]:
            refuse("%s format %r" % (kind, fmt))
        return ref + "\n"
    refuse("inspect type " + kind)


def d_network(state, a):
    if a[:3] != ["create", "--driver", "bridge"] or len(a) != 4:
        refuse("network shape %r (only: create --driver bridge NAME; not internal)" % a)
    if a[3] in state["networks"]:
        fail("Error response from daemon: network with name %s already exists" % a[3])
    state["networks"][a[3]] = {"driver": "bridge", "internal": False}
    save(state)
    return new_id(state, "network") + "\n"


def d_volume(state, a):
    if len(a) != 2 or a[0] != "create":
        refuse("volume shape %r (a volume is only ever created)" % a)
    state["volumes"].setdefault(a[1], {"files": {}})
    save(state)
    return a[1] + "\n"


def d_build(state, a):
    tag = fil = ctx = None
    args = {}
    i = 0
    while i < len(a):
        x = a[i]
        if x in ("--tag", "--file", "--build-arg") and i + 1 < len(a):
            if x == "--tag":
                tag = a[i + 1]
            elif x == "--file":
                fil = a[i + 1]
            else:
                k, _, v = a[i + 1].partition("=")
                args[k] = v
            i += 2
        elif x.startswith("-") and x != "-":
            refuse("build flag " + x)
        else:
            if ctx is not None:
                refuse("two build contexts")
            ctx = x
            i += 1
    m = re.fullmatch(r"raphael-journeys-(proxy|app|browser|platform):([A-Za-z0-9][A-Za-z0-9._-]*)", tag or "")
    if not m or ctx is None:
        refuse("build tag or context %r %r" % (tag, ctx))
    role, version = m.groups()
    built = {"file": fil, "args": args, "tag": tag, "context_files": [], "stdin_bytes": 0}
    env = {"PATH": "/usr/local/bin:/usr/bin"}
    health = None
    if role == "platform":
        if ctx != "-" or fil is not None or set(args) != {"HERMES_GIT_SHA"} or args["HERMES_GIT_SHA"] != version:
            refuse("platform build shape")
        built["stdin_bytes"] = len(sys.stdin.buffer.read())
        env["HERMES_HOME"] = "/opt/data"
    else:
        folder = Path(ctx)
        if not folder.is_dir() or not fil or not Path(fil).is_file() or Path(fil).parent.name != "images":
            refuse("build needs a context folder and a Dockerfile from the images folder")
        built["context_files"] = sorted(p.name for p in folder.iterdir())
        if Path(fil).name != role + ("-ca" if role == "app" else "") + ".Dockerfile":
            refuse("Dockerfile %s does not belong to the %s image" % (Path(fil).name, role))
        if role == "proxy":
            if built["context_files"] != ["synthetic_proxy.py"] or args:
                refuse("proxy build context or arguments")
        else:
            if built["context_files"] != ["ca.pem"]:
                refuse("%s build context must hold ca.pem only" % role)
            suffix = hashlib.sha256((folder / "ca.pem").read_bytes()).hexdigest()[:12]
            if not version.endswith("-" + suffix):
                refuse("%s tag does not end with the hash of the CA in its build folder" % role)
            if role == "app":
                _, base = find_image(state, args.get("APP_IMAGE_DIGEST", ""))
                if set(args) != {"APP_IMAGE_DIGEST"} or base is None:
                    refuse("app build needs APP_IMAGE_DIGEST naming the released image")
                env, health = dict(base["env"]), base["health"]
            elif args:
                refuse("browser build arguments")
    image_id = "sha256:" + hashlib.sha256(("%s-%d" % (tag, state["seq"] + 1)).encode()).hexdigest()
    state["seq"] += 1
    state["images"][tag] = {"id": image_id, "env": env, "health": health, "built": built}
    save(state)
    return ""


def parse_mount(state, text):
    parts = dict(p.split("=", 1) for p in text.split(",") if "=" in p)
    if set(parts) != {"type", "source", "target"} or parts["type"] != "volume":
        refuse("mount %r (volume mounts only: type,source,target)" % text)
    if parts["source"] not in state["volumes"]:
        fail("docker: Error response from daemon: no such volume: " + parts["source"], 125)
    return {"name": parts["source"], "dest": parts["target"]}


VALUE_FLAGS = {"--name", "--label", "--restart", "--network", "--network-alias", "--mount", "--env-file",
               "--health-cmd", "--health-interval", "--health-timeout", "--health-retries",
               "--health-start-period", "--user", "--entrypoint"}
BOOL_FLAGS = {"--detach", "--rm", "-i"}
PORT_FLAGS = {"-p", "--publish", "-P", "--publish-all", "--expose"}


def d_run(state, a):
    opts = {}
    i = 0
    while i < len(a) and a[i].startswith("-"):
        f = a[i]
        if f in PORT_FLAGS:
            refuse("no port may be published (%s)" % f)
        if f in BOOL_FLAGS:
            opts.setdefault(f, []).append(True)
            i += 1
        elif f in VALUE_FLAGS and i + 1 < len(a):
            opts.setdefault(f, []).append(a[i + 1])
            i += 2
        else:
            refuse("run flag %s (not expected; docker also refuses --hostname, --dns, --add-host and links "
                   "with --network container:)" % f)
    if i >= len(a):
        refuse("run without an image")
    image_ref, command = a[i], a[i + 1:]
    _, img = find_image(state, image_ref)
    if img is None:
        fail("docker: Unable to find image '%s' locally" % image_ref, 125)
    if opts.get("--rm"):
        if set(opts) - {"--rm", "-i", "--network", "--user", "--entrypoint", "--mount"}:
            refuse("helper run flags")
        if opts.get("--network") != ["none"] or opts.get("--user") != ["0"] or opts.get("--entrypoint") != ["sh"]:
            refuse("helper run needs --network none --user 0 --entrypoint sh")
        if len(opts.get("--mount", [])) != 1 or len(command) != 2 or command[0] != "-c":
            refuse("helper run shape")
        parse_mount(state, opts["--mount"][0])
        if opts.get("-i"):
            sys.stdin.buffer.read()
        return ""
    for f in ("--detach", "--name", "--label", "--restart", "--network"):
        if len(opts.get(f, [])) != (0 if f == "--detach" else 1) and f != "--detach":
            refuse("run needs exactly one %s" % f)
    if not opts.get("--detach"):
        refuse("run without --detach")
    if opts["--label"] != [LABEL_KEY + "=stack_up"]:
        refuse("run label %r" % opts["--label"])
    if opts["--restart"] != ["unless-stopped"]:
        refuse("restart policy %r" % opts["--restart"])
    if set(opts) & {"--user", "--entrypoint"}:
        refuse("--user and --entrypoint are not expected")
    name, net = opts["--name"][0], opts["--network"][0]
    aliases = opts.get("--network-alias", [])
    if any(c["name"] == name for c in state["containers"].values()):
        fail('docker: Error response from daemon: Conflict. The container name "/%s" is already in use' % name, 125)
    if net.startswith("container:"):
        if aliases:
            refuse("a network alias has no endpoint to attach to with --network container:")
        target = net[len("container:"):]
        if not HEX64.fullmatch(target):
            refuse("--network container: must name the container by its id")
        _, t = find_container(state, target)
        if t is None or not t["running"]:
            fail("docker: Error response from daemon: cannot join network of a non running container: " + target, 125)
        if t["labels"].get(LABEL_KEY) != "stack_up":
            refuse("joining the namespace of a container the script did not make")
    elif net not in state["networks"]:
        fail("docker: Error response from daemon: network %s not found" % net, 125)
    env = {}
    if "--env-file" in opts:
        path = Path(opts["--env-file"][0])
        if not path.is_file():
            fail("docker: open %s: no such file or directory" % path, 125)
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            key, eq, value = line.partition("=")
            if not eq or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
                refuse("env file line without NAME=value (a bare name would pass a host value through)")
            env[key] = value
    mounts = [parse_mount(state, m) for m in opts.get("--mount", [])]
    health = img["health"]
    if "--health-cmd" in opts:
        health = ["CMD-SHELL", opts["--health-cmd"][0]]
    cid = new_id(state, "container")
    state["containers"][cid] = {
        "id": cid, "name": name, "image_ref": image_ref, "image_id": img["id"], "labels": {LABEL_KEY: "stack_up"},
        "env": env, "mounts": mounts, "network": net, "aliases": aliases, "ports": [], "restart": "unless-stopped",
        "health": health, "health_opts": {k: v[0] for k, v in opts.items() if k.startswith("--health-")},
        "cmd": command, "running": True, "status": "starting", "polls": state["polls"],
    }
    if image_ref.startswith("raphael-journeys-proxy:"):
        for m in mounts:
            files = state["volumes"][m["name"]]["files"]
            files.setdefault("ca.pem", "-----BEGIN CERTIFICATE-----\nstand-in-ca-%d\n-----END CERTIFICATE-----\n" % state["seq"])
            files.setdefault("ca-key.pem", "stand-in private key, never leaves the volume\n")
    save(state)
    return cid + "\n"


def d_cp(state, a):
    if len(a) != 2 or ":" not in a[0]:
        refuse("cp shape %r (copy out of a container only)" % a)
    cid, _, path = a[0].partition(":")
    if not HEX64.fullmatch(cid):
        refuse("cp must name the container by its id")
    _, c = find_container(state, cid)
    if c is None:
        fail("Error: No such container:path: " + a[0])
    for m in c["mounts"]:
        if path == m["dest"].rstrip("/") + "/ca.pem" and m["name"] == "raphael-journeys-proxy":
            dest = Path(a[1])
            if dest.is_dir():
                dest = dest / "ca.pem"
            dest.write_text(state["volumes"][m["name"]]["files"]["ca.pem"], encoding="utf-8")
            return ""
    refuse("cp of %s (only ca.pem may leave the proxy)" % path)


def d_rm(state, a):
    if len(a) != 2 or a[0] != "--force" or not HEX64.fullmatch(a[1]):
        refuse("rm shape %r (only: rm --force <container id>; never by name, never -v)" % a)
    if a[1] not in state["containers"]:
        fail("Error response from daemon: No such container: " + a[1])
    del state["containers"][a[1]]
    save(state)
    return a[1] + "\n"


def d_lifecycle(state, a, sub):
    if len(a) != 1:
        refuse("%s shape %r" % (sub, a))
    cid, c = find_container(state, a[0])
    if c is None:
        fail("Error response from daemon: No such container: " + a[0])
    if sub in ("stop", "restart"):
        c["running"] = False
    if sub in ("start", "restart"):
        if not c["running"]:
            join_check(state, c)
        c["running"], c["status"], c["polls"] = True, "starting", state["polls"]
    save(state)
    return a[0] + "\n"


def d_exec(state, a):
    i, interactive = 0, False
    while i < len(a) and a[i].startswith("-"):
        if a[i] == "-i":
            interactive = True
            i += 1
        elif a[i] == "--user" and i + 1 < len(a):
            i += 2
        else:
            refuse("exec flag " + a[i])
    _, c = find_container(state, a[i])
    if c is None:
        fail("Error response from daemon: No such container: " + a[i])
    if not c["running"]:
        fail("Error response from daemon: container is not running")
    stdin = sys.stdin.buffer.read().decode() if interactive else ""
    cmd = a[i + 1:]
    if cmd in (["id", "-u"], ["id", "-g"]):
        return "1000\n"
    if cmd[:3] == ["stat", "-c", "%u:%g"] and len(cmd) == 4:
        return "10000:10000\n"
    if len(cmd) == 2 and cmd[0] == "printenv":
        env = merged_env(state, c)
        if cmd[1] not in env:
            raise Done(1)
        return env[cmd[1]] + "\n"
    if cmd[:3] == ["hermes", "kanban-workspace-token", "issue"]:
        return ""
    if cmd[:2] == ["sh", "-c"] and len(cmd) >= 3:
        script = cmd[2]
        if 'printf "%s %s\\n" "$w" "$n"' in script:
            return "default API_SERVER_KEY\nraphael-planner API_SERVER_KEY\n"
        if 'printf "entry %s\\n"' in script:
            return "entry raphael-planner\n"
        if 'exec "$@" -' in script and "profiles_to_serve" in stdin:
            return "served default\nserved raphael-planner\n"
        if "head -c 32 /dev/urandom" in script or 'cat >> "$1"' in script:
            return ""
    refuse("exec program %r" % (cmd[:3],))


def run_docker(argv):
    state = load()
    if not argv:
        refuse("no subcommand")
    sub, rest = argv[0], argv[1:]
    table = {"info": d_info, "inspect": d_inspect, "network": d_network, "volume": d_volume, "build": d_build,
             "run": d_run, "cp": d_cp, "rm": d_rm, "exec": d_exec}
    if sub in table:
        return table[sub](state, rest)
    if sub in ("start", "stop", "restart"):
        return d_lifecycle(state, rest, sub)
    refuse("docker subcommand %r" % sub)


def main():
    global TOOL
    TOOL, argv = sys.argv[1], sys.argv[2:]
    rc, out = 0, ""
    try:
        if TOOL == "docker":
            out = run_docker(argv) or ""
        elif TOOL == "hostname":
            out = load()["host"] + "\n"
        elif TOOL == "df":
            state = load()
            if len(argv) != 2 or argv[0] != "-Pk" or argv[1] != state["docker_root"]:
                refuse("df must be: df -Pk <docker data folder>")
            out = "Filesystem 1024-blocks Used Available Capacity Mounted on\nstand-in %d 1 %d 1%% /\n" % (
                state["free_kb"] + 1, state["free_kb"])
        elif TOOL != "sleep":
            refuse("tool")
    except Done as done:
        rc = done.rc
        sys.stderr.write(done.err)
    sys.stdout.write(out)
    log(argv, rc)
    sys.exit(rc)


main()
'''

PROXY_TAG = "0123456789ab"
IMAGES_DIR_NAME = "images"


def parse_template() -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {}
    current = None
    for raw in TEMPLATE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            current = line[1:-1]
            sections[current] = []
        else:
            sections[current].append(line)
    return sections


SECTIONS = parse_template()
SETTINGS = dict(
    line.split("=", 1)
    for line in (JOURNEYS / "stack-app.settings").read_text(encoding="utf-8").splitlines()
    if line.strip() and not line.startswith("#")
)


def seed_image(state: dict, ref: str, env: dict | None = None, health: list | None = None) -> None:
    state["seq"] += 1
    state["images"][ref] = {
        "id": "sha256:" + hashlib.sha256(ref.encode()).hexdigest(),
        "env": env or {"PATH": "/usr/local/bin:/usr/bin"},
        "health": health,
        "built": {"file": None, "args": {}, "tag": ref, "context_files": [], "stdin_bytes": 0},
    }


def fresh_state() -> dict:
    state = {
        "host": HOST,
        "seq": 0,
        "polls": 2,
        "free_kb": 100 * 1024 * 1024,
        "docker_root": "/var/lib/stand-in-docker",
        "security": ["name=seccomp,profile=builtin", "name=rootless", "name=cgroupns"],
        "networks": {},
        "volumes": {},
        "containers": {},
        "images": {},
        "never_healthy": [],
    }
    seed_image(state, f"raphael-journeys-proxy:{PROXY_TAG}")
    seed_image(state, f"raphael-journeys-platform:{PLATFORM}", env={"PATH": "/usr/bin", "HERMES_HOME": "/opt/data"})
    for revision in (REVISION, NEW_REVISION):
        seed_image(
            state,
            f"raphael-journeys-app-base:{revision}",
            env={"PATH": "/usr/bin", "NODE_VERSION": "22"},
            health=["CMD", "node", "healthcheck.js"],
        )
    return state


class Host:
    """A fake Server 2: a copy of scripts/journeys, the stand-ins first on PATH, and the fake host state."""

    def __init__(self, root: Path, **overrides):
        self.root = root
        self.scripts = root / "tree" / "scripts" / "journeys"
        shutil.copytree(JOURNEYS, self.scripts)
        self.bin = root / "bin"
        self.bin.mkdir()
        (self.bin / "stub.py").write_text(f"F = {FORMATS!r}\n" + STUB, encoding="utf-8")
        for tool in ("docker", "sleep", "hostname", "df"):
            wrapper = self.bin / tool
            wrapper.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$(dirname "$0")/stub.py" {tool} "$@"\n', encoding="utf-8")
            wrapper.chmod(0o755)
        (root / "home").mkdir()
        (root / "tmp").mkdir()
        state = fresh_state()
        state.update(overrides)
        self.save(state)
        (root / "calls.jsonl").write_text("", encoding="utf-8")

    def env(self, **extra: str) -> dict[str, str]:
        env = {
            "PATH": f"{self.bin}:/usr/bin:/bin",
            "HOME": str(self.root / "home"),
            "TMPDIR": str(self.root / "tmp"),
            "DOCKER_HOST": "unix:///nonexistent/stand-in-only/docker.sock",
            "JOURNEY_STACK_HOST": HOST,
            "LC_ALL": "C",
        }
        env.update(extra)
        return env

    def run(self, script: str, *args: str, timeout: int = 120, **extra_env: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [str(self.scripts / script), *args],
            env=self.env(**extra_env),
            cwd=self.root,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    def converge(self, app: str = REVISION, platform: str = PLATFORM, **extra_env: str) -> subprocess.CompletedProcess:
        return self.run("stack_up.sh", "converge", PROXY_TAG, app, platform, app, **extra_env)

    def state(self) -> dict:
        return json.loads((self.root / "state.json").read_text(encoding="utf-8"))

    def save(self, state: dict) -> None:
        (self.root / "state.json").write_text(json.dumps(state), encoding="utf-8")

    def calls(self) -> list[dict]:
        text = (self.root / "calls.jsonl").read_text(encoding="utf-8")
        return [json.loads(line) for line in text.splitlines()]

    def docker_calls(self, since: int = 0) -> list[list[str]]:
        return [c["argv"] for c in self.calls()[since:] if c["tool"] == "docker"]

    def mark(self) -> int:
        return len(self.calls())

    def containers(self) -> dict[str, dict]:
        return {c["name"]: c for c in self.state()["containers"].values()}


def changing(docker_calls: list[list[str]]) -> list[list[str]]:
    return [argv for argv in docker_calls if argv[0] not in ("info", "inspect")]


def output_lines(result: subprocess.CompletedProcess) -> list[str]:
    return (result.stdout + result.stderr).splitlines()


def run_names(docker_calls: list[list[str]]) -> list[str]:
    names = []
    for argv in docker_calls:
        if argv[0] == "run" and "--name" in argv:
            names.append(argv[argv.index("--name") + 1])
    return names


@pytest.fixture
def host(tmp_path: Path) -> Host:
    return Host(tmp_path)


@pytest.fixture
def stack(host: Host) -> Host:
    result = host.converge()
    assert result.returncode == 0, result.stdout + result.stderr
    return host


def proxy_ca_pem(state: dict) -> str:
    return state["volumes"][PROXY_V]["files"]["ca.pem"]


def ca_part(state: dict) -> str:
    return hashlib.sha256(proxy_ca_pem(state).encode()).hexdigest()[:12]


def test_1_empty_host_creates_network_volumes_and_five_containers_in_order(host: Host) -> None:
    """Empty host: the section 2 order, the shape of each container, and decisions 1 to 5 and 7."""
    result = host.converge()
    assert result.returncode == 0, result.stdout + result.stderr
    assert output_lines(result)[-1] == "stack_up: the stack answers"

    docker = host.docker_calls()
    made = changing(docker)
    assert made[0] == ["network", "create", "--driver", "bridge", NETWORK]
    assert made[1:4] == [["volume", "create", name] for name in STACK_VOLUMES]
    assert [argv[0] for argv in made] == (
        ["network", "volume", "volume", "volume", "run", "cp", "build", "build", "run", "run", "run", "run"]
    )
    assert run_names(docker) == [PROXY_C, DASH_C, GATE_C, APP_C, BROWSER_C]

    state = host.state()
    containers = host.containers()
    assert sorted(containers) == sorted(STACK_CONTAINERS)
    for name, container in containers.items():
        assert container["health"] and container["health"][0] in ("CMD", "CMD-SHELL"), name
        assert all(m["name"] in SECTIONS["volumes"] for m in container["mounts"]), name
        assert container["ports"] == [], name
        assert container["restart"] == "unless-stopped", name
        assert container["labels"] == {"raphael-journeys.made-by": "stack_up"}, name
        allowed = set(SECTIONS[f"env {name}"]) | set(state["images"][container["image_ref"]]["env"])
        assert set(container["env"]) <= allowed, name
    for argv in docker:
        assert not {"-p", "--publish", "-P", "--publish-all", "--expose"} & set(argv)
        if argv[0] == "run":
            assert argv[argv.index("--label") + 1] == LABEL
    assert state["networks"][NETWORK] == {"driver": "bridge", "internal": False}
    assert sorted(state["volumes"]) == sorted(STACK_VOLUMES)

    # Decision 5: images from scripts/journeys/images, or the released owner app image plus the CA layer.
    images_dir = (host.scripts / IMAGES_DIR_NAME).resolve()
    ca = ca_part(state)
    assert containers[PROXY_C]["image_ref"] == f"raphael-journeys-proxy:{PROXY_TAG}"
    assert containers[DASH_C]["image_ref"] == containers[GATE_C]["image_ref"] == f"raphael-journeys-platform:{PLATFORM}"
    app_built = state["images"][containers[APP_C]["image_ref"]]["built"]
    browser_built = state["images"][containers[BROWSER_C]["image_ref"]]["built"]
    assert containers[APP_C]["image_ref"] == f"raphael-journeys-app:{REVISION}-{ca}"
    assert containers[BROWSER_C]["image_ref"] == f"raphael-journeys-browser:{REVISION}-{ca}"
    assert Path(app_built["file"]).resolve() == images_dir / "app-ca.Dockerfile"
    assert app_built["args"] == {"APP_IMAGE_DIGEST": f"raphael-journeys-app-base:{REVISION}"}
    assert Path(browser_built["file"]).resolve() == images_dir / "browser.Dockerfile"

    # Decision 1: the owner app joins the gateway container's network; the aliases sit on the gateway.
    gateway_id = next(cid for cid, c in state["containers"].items() if c["name"] == GATE_C)
    assert containers[APP_C]["network"] == f"container:{gateway_id}"
    assert containers[APP_C]["aliases"] == []
    assert sorted(containers[GATE_C]["aliases"]) == ["gateway", "workspace"]
    assert containers[DASH_C]["aliases"] == ["dashboard"]
    assert sorted(containers[PROXY_C]["aliases"]) == ["synthetic-team.cloudflareaccess.com", "workspace.synthetic.test"]
    for name in (PROXY_C, DASH_C, GATE_C, BROWSER_C):
        assert containers[name]["network"] == NETWORK

    # Decision 2: exactly one gateway, started by the platform's own decision function; no dashboard anywhere else.
    from hermes_cli.container_boot import _is_dashboard_container, _is_legacy_gateway_run_request

    def booted(name: str) -> list[str]:
        return ["/init", "/opt/hermes/docker/main-wrapper.sh", "hermes", *containers[name]["cmd"]]

    gateways = [name for name in containers if containers[name]["cmd"][:2] == ["gateway", "run"]]
    assert gateways == [GATE_C]
    for name in containers:
        assert not _is_legacy_gateway_run_request(booted(name)), name
    assert not _is_dashboard_container(booted(GATE_C))
    assert _is_dashboard_container(booted(DASH_C))
    for name, container in containers.items():
        image_env = state["images"][container["image_ref"]]["env"]
        assert "HERMES_DASHBOARD" not in {**image_env, **container["env"]}, name

    # Decision 3: the dashboard command and its /api/health check.
    assert containers[DASH_C]["cmd"] == ["dashboard", "--host", "0.0.0.0", "--port", "9119", "--no-open"]
    assert containers[DASH_C]["health"][0] == "CMD-SHELL"
    assert "127.0.0.1:9119/api/health" in containers[DASH_C]["health"][1]
    assert "127.0.0.1:8642/health" in containers[GATE_C]["health"][1]


def test_2_reset_stack_guards_all_pass_on_the_created_stack(stack: Host) -> None:
    """reset_stack.sh against the same state: every guard passes and its first change is reached."""
    mark = stack.mark()
    result = stack.run("reset_stack.sh")
    text = result.stdout + result.stderr
    assert "nothing was changed" not in text, text
    assert "reset_stack: stopping " + GATE_C in text, text
    stops = [argv for argv in stack.docker_calls(mark) if argv[0] == "stop"]
    assert stops and stops[0] == ["stop", GATE_C]


def test_3_isolation_check_names_on_template_passes_on_the_created_stack(stack: Host) -> None:
    """isolation_check.sh server2: the names-on-template check passes against the same state."""
    result = stack.run("isolation_check.sh", "server2")
    assert "PASS names-on-template" in result.stdout.splitlines(), result.stdout + result.stderr


def test_4_second_converge_with_the_same_tags_makes_no_create_remove_or_volume_call(stack: Host) -> None:
    """A second converge with the same tags changes nothing."""
    before = {c["name"]: cid for cid, c in stack.state()["containers"].items()}
    mark = stack.mark()
    result = stack.converge()
    assert result.returncode == 0, result.stdout + result.stderr
    calls = stack.docker_calls(mark)
    assert {argv[0] for argv in calls} <= {"info", "inspect", "cp"}, calls
    assert {c["name"]: cid for cid, c in stack.state()["containers"].items()} == before
    assert output_lines(result)[-1] == "stack_up: the stack answers"


def container_ids(host: Host) -> dict[str, str]:
    return {c["name"]: cid for cid, c in host.state()["containers"].items()}


def test_5_new_owner_app_tag_recreates_only_the_owner_app_and_the_browser(stack: Host) -> None:
    """A new owner app tag: two removals by id, two creations, no volume call, no proxy volume in a remove."""
    before = container_ids(stack)
    mark = stack.mark()
    result = stack.converge(app=NEW_REVISION)
    assert result.returncode == 0, result.stdout + result.stderr
    docker = stack.docker_calls(mark)

    removed = [argv for argv in docker if argv[0] == "rm"]
    assert sorted(argv[2] for argv in removed) == sorted([before[APP_C], before[BROWSER_C]])
    assert all(argv[:2] == ["rm", "--force"] and len(argv) == 3 for argv in removed)
    assert sorted(run_names(docker)) == sorted([APP_C, BROWSER_C])
    assert not [argv for argv in docker if argv[0] in ("volume", "network")]

    after = container_ids(stack)
    for name in (PROXY_C, DASH_C, GATE_C):
        assert after[name] == before[name], name
    for name in (APP_C, BROWSER_C):
        assert after[name] != before[name], name
    state = stack.state()
    ca = ca_part(state)
    assert containers_by_name(state)[APP_C]["image_ref"] == f"raphael-journeys-app:{NEW_REVISION}-{ca}"
    assert containers_by_name(state)[APP_C]["network"] == f"container:{before[GATE_C]}"
    assert sorted(state["volumes"]) == sorted(STACK_VOLUMES)

    for argv in stack.docker_calls():
        assert argv[0] not in ("system", "image", "rmi", "prune")
        if argv[0] == "volume":
            assert argv[1] == "create"
        if argv[0] == "rm":
            assert not {"-v", "--volumes", PROXY_V, PROXY_C} & set(argv)
            assert re.fullmatch(r"[0-9a-f]{64}", argv[-1])


def containers_by_name(state: dict) -> dict[str, dict]:
    return {c["name"]: c for c in state["containers"].values()}


def foreign_container(state: dict, name: str) -> str:
    state["seq"] += 1
    cid = hashlib.sha256(f"foreign-{name}".encode()).hexdigest()
    image_ref = f"raphael-journeys-proxy:{PROXY_TAG}"
    state["containers"][cid] = {
        "id": cid, "name": name, "image_ref": image_ref, "image_id": state["images"][image_ref]["id"],
        "labels": {}, "env": {}, "mounts": [], "network": "bridge", "aliases": [], "ports": [],
        "restart": "no", "health": None, "health_opts": {}, "cmd": [], "running": True, "status": "none", "polls": 0,
    }
    return cid


@pytest.mark.parametrize("case", ["wrong_host", "tag_not_plain", "unlabelled_container", "less_than_20_gb_free"])
def test_6_refusals_change_nothing(tmp_path: Path, case: str) -> None:
    """Each refusal exits non-zero before any call that changes anything, and the fake host is unchanged."""
    overrides: dict = {}
    if case == "wrong_host":
        overrides["host"] = "some-other-host.invalid"
    if case == "less_than_20_gb_free":
        overrides["free_kb"] = 19 * 1024 * 1024
    host = Host(tmp_path, **overrides)
    if case == "unlabelled_container":
        state = host.state()
        foreign_container(state, PROXY_C)
        host.save(state)
    before = host.state()
    if case == "tag_not_plain":
        result = host.run("stack_up.sh", "converge", PROXY_TAG, "../../etc/passwd", PLATFORM, REVISION)
    else:
        result = host.converge()
    assert result.returncode != 0
    assert "nothing was changed" in result.stderr
    assert changing(host.docker_calls()) == []
    assert host.state() == before


def test_7_ca_leaves_the_proxy_only_as_ca_pem_and_only_the_proxy_mounts_the_proxy_volume(stack: Host) -> None:
    """One docker cp of ca.pem; the build folders hold ca.pem only; no other container mounts the proxy volume."""
    state = stack.state()
    proxy_id = container_ids(stack)[PROXY_C]
    docker = stack.docker_calls()
    copies = [argv for argv in docker if argv[0] == "cp"]
    assert len(copies) == 1
    assert copies[0][1] == f"{proxy_id}:/state/ca.pem" and Path(copies[0][2]).name == "ca.pem"
    layers = [img for ref, img in state["images"].items() if ref.startswith(("raphael-journeys-app:", "raphael-journeys-browser:"))]
    assert len(layers) == 2
    assert all(img["built"]["context_files"] == ["ca.pem"] for img in layers)
    mounting = [c["name"] for c in state["containers"].values() if any(m["name"] == PROXY_V for m in c["mounts"])]
    assert mounting == [PROXY_C]
    for argv in docker:
        if argv[0] == "run":
            mounts = [argv[i + 1] for i, word in enumerate(argv) if word == "--mount"]
            names_proxy_volume = any(f"source={PROXY_V}," in m for m in mounts)
            assert names_proxy_volume == (argv[argv.index("--name") + 1] == PROXY_C)


def test_8_no_output_line_or_command_line_carries_an_environment_value(host: Host) -> None:
    """Values reach docker only through --env-file; neither the output nor any command line shows one."""
    results = [host.converge(), host.converge(app=NEW_REVISION)]
    results.append(host.run("stack_up.sh", "converge", PROXY_TAG, "bad tag", PLATFORM, REVISION))
    results.append(host.run("stack_up.sh", "versions"))
    state = host.state()
    app_env = containers_by_name(state)[APP_C]["env"]
    assert app_env == SETTINGS
    values = {value for value in SETTINGS.values() if value} | {"/state", "/opt/data"}
    for result in results:
        for line in output_lines(result):
            assert not [v for v in values if v in line], line
    for argv in host.docker_calls():
        for word in argv:
            if word == "--mount" or word.startswith("type=volume,"):
                continue
            assert not [v for v in values - {"/state", "/opt/data"} if v in word and len(v) > 4], word
        assert "-e" not in argv and "--env" not in argv
        if argv[0] == "run" and argv[argv.index("--name") + 1] != BROWSER_C:
            assert "--env-file" in argv


# --- extras: not among the plan's eight tests ---


def sub_host(parent: Path, name: str, **overrides) -> Host:
    root = parent / name
    root.mkdir()
    return Host(root, **overrides)


def test_extra_versions_prints_one_line_per_role_and_changes_nothing(tmp_path: Path) -> None:
    """versions: five "<role> <tag>" lines in template order; every tag is none on an empty host."""
    empty = sub_host(tmp_path, "empty")
    result = empty.run("stack_up.sh", "versions")
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [f"{role} none" for role in ("proxy", "app", "dashboard", "gateway", "browser")]
    assert changing(empty.docker_calls()) == []

    made = sub_host(tmp_path, "made")
    assert made.converge().returncode == 0
    mark = made.mark()
    result = made.run("stack_up.sh", "versions")
    ca = ca_part(made.state())
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        f"proxy {PROXY_TAG}",
        f"app {REVISION}-{ca}",
        f"dashboard {PLATFORM}",
        f"gateway {PLATFORM}",
        f"browser {REVISION}-{ca}",
    ]
    assert changing(made.docker_calls(mark)) == []


def test_extra_build_proxy_makes_one_image_tagged_by_the_hash_of_its_inputs(host: Host) -> None:
    """build proxy: one docker build, tag = first 12 hex of sha256 over synthetic_proxy.py then proxy.Dockerfile."""
    result = host.run("stack_up.sh", "build", "proxy")
    assert result.returncode == 0, result.stdout + result.stderr
    digest = hashlib.sha256()
    digest.update((JOURNEYS / "synthetic_proxy.py").read_bytes())
    digest.update((JOURNEYS / "images" / "proxy.Dockerfile").read_bytes())
    tag = digest.hexdigest()[:12]
    assert result.stdout.splitlines() == [f"stack_up: built proxy {tag}"]
    builds = [argv for argv in host.docker_calls() if argv[0] == "build"]
    assert len(builds) == 1 and builds[0][builds[0].index("--tag") + 1] == f"raphael-journeys-proxy:{tag}"
    assert changing(host.docker_calls()) == builds
    assert host.state()["images"][f"raphael-journeys-proxy:{tag}"]["built"]["context_files"] == ["synthetic_proxy.py"]


def wait_for_image(host: Host, ref: str) -> dict:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        images = host.state()["images"]
        if ref in images:
            return images[ref]
        time.sleep(0.2)
    raise AssertionError(f"{ref} was never built")


def test_extra_build_platform_returns_at_once_builds_detached_and_is_not_repeated(host: Host) -> None:
    """build platform: one line and exit 0 at once; the detached build reads the archive; a second call is a no-op."""
    archive = host.root / "platform.tar"
    archive.write_bytes(b"archive" * 100)
    ref = f"raphael-journeys-platform:{NEW_PLATFORM}"
    result = host.run("stack_up.sh", "build", "platform", NEW_PLATFORM, str(archive))
    assert result.returncode == 0, result.stdout + result.stderr
    log_file = host.root / "tmp" / "stack_up-platform-build.log"
    assert result.stdout.splitlines() == [f"stack_up: building platform {NEW_PLATFORM}; log {log_file}"]
    built = wait_for_image(host, ref)["built"]
    assert built["stdin_bytes"] == 700 and built["args"] == {"HERMES_GIT_SHA": NEW_PLATFORM}
    mark = host.mark()
    again = host.run("stack_up.sh", "build", "platform", NEW_PLATFORM, str(archive))
    assert again.returncode == 0 and again.stdout.splitlines() == [f"stack_up: image {ref} already exists"]
    assert not [argv for argv in host.docker_calls(mark) if argv[0] == "build"]


def test_extra_build_platform_leaves_a_running_build_alone(host: Host) -> None:
    """While another build holds the lock, build platform starts nothing and says so."""
    import fcntl

    archive = host.root / "platform.tar"
    archive.write_bytes(b"archive")
    with (host.root / "tmp" / "stack_up-platform-build.lock").open("wb") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = host.run("stack_up.sh", "build", "platform", NEW_PLATFORM, str(archive))
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.splitlines() == ["stack_up: a platform build is already running"]
    assert not [argv for argv in host.docker_calls() if argv[0] == "build"]


def test_extra_docker_that_is_not_rootless_is_refused_before_any_change(tmp_path: Path) -> None:
    host = Host(tmp_path, security=["name=seccomp,profile=builtin"])
    before = host.state()
    result = host.converge()
    assert result.returncode != 0 and "not rootless" in result.stderr and "nothing was changed" in result.stderr
    assert changing(host.docker_calls()) == [] and host.state() == before


def test_extra_script_is_mode_755_and_wrong_usage_exits_2_before_any_docker_call(host: Host) -> None:
    assert (JOURNEYS / "stack_up.sh").stat().st_mode & 0o777 == 0o755
    for args in ([], ["build"], ["build", "nothing"], ["build", "app"], ["converge", PROXY_TAG], ["versions", "x"]):
        result = host.run("stack_up.sh", *args)
        assert result.returncode == 2, args
        assert result.stderr.startswith("usage: stack_up.sh"), args
    assert host.docker_calls() == []


def test_extra_new_platform_commit_recreates_dashboard_gateway_then_owner_app_in_order(stack: Host) -> None:
    """A new platform commit: dashboard, owner app and gateway are removed by id, and re-created gateway first."""
    state = stack.state()
    seed_image(state, f"raphael-journeys-platform:{NEW_PLATFORM}", env={"PATH": "/usr/bin", "HERMES_HOME": "/opt/data"})
    stack.save(state)
    before = container_ids(stack)
    mark = stack.mark()
    result = stack.converge(platform=NEW_PLATFORM)
    assert result.returncode == 0, result.stdout + result.stderr
    docker = stack.docker_calls(mark)
    removed = [argv[2] for argv in docker if argv[0] == "rm"]
    assert sorted(removed) == sorted([before[DASH_C], before[GATE_C], before[APP_C]])
    assert removed.index(before[APP_C]) < removed.index(before[GATE_C])
    assert run_names(docker) == [DASH_C, GATE_C, APP_C]
    after = container_ids(stack)
    assert after[PROXY_C] == before[PROXY_C] and after[BROWSER_C] == before[BROWSER_C]
    assert containers_by_name(stack.state())[APP_C]["network"] == f"container:{after[GATE_C]}"
    assert not [argv for argv in docker if argv[0] in ("volume", "network")]


def test_extra_container_that_never_answers_fails_in_one_budget_and_converge_resumes(tmp_path: Path) -> None:
    """No answer: exit 1 after at most 90 polls of 2 seconds; fixing the cause and converging again makes no change."""
    host = Host(tmp_path, never_healthy=[BROWSER_C])
    result = host.converge()
    assert result.returncode == 1 and "run converge again" in result.stderr
    sleeps = [c["argv"] for c in host.calls() if c["tool"] == "sleep"]
    assert 0 < len(sleeps) <= 90 and all(argv == ["2"] for argv in sleeps)
    assert sorted(host.containers()) == sorted(STACK_CONTAINERS)
    state = host.state()
    state["never_healthy"] = []
    host.save(state)
    mark = host.mark()
    result = host.converge()
    assert result.returncode == 0, result.stdout + result.stderr
    assert {argv[0] for argv in host.docker_calls(mark)} <= {"info", "inspect", "cp"}
