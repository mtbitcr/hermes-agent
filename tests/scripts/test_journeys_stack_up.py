"""Tests for scripts/journeys/stack_up.sh (card 2 of the journeys stack plan, round 2).

The script runs against stand-ins for docker, sleep, hostname and df that sit first on PATH. The stand-in
docker records every call in calls.jsonl, keeps a fake Server 2 in state.json, and refuses (status 98) any
subcommand or flag the script is not expected to use. It also models what the real engine and the real owner
app do that matter here: it refuses to start a container that joins the namespace of a container that is not
running, refuses a --network container: container that also asks for an alias, hostname, published port, dns,
add-host, expose or link, keeps the files of each volume with their sizes, and makes a started owner app exit
at once when one of the seven token files of stack-app.settings is missing or empty. DOCKER_HOST points at a
path that does not exist, so a real docker could not answer. No real Docker call is made anywhere. The tests
read recorded calls, output lines and the fake host state; they never read the script's text, and the token
files hold invented placeholder text that is never printed.

Tests test_1_... to test_8_... are the eight tests of the plan (test_1 and test_2 cover the round 2 rules);
test_9_... onwards and test_extra_... are additions.
"""

from __future__ import annotations

import argparse
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
            shown = "none" if not c["health"] or c["status"] in ("created", "exited") else c["status"]
            out = "%s %s" % ("true" if c["running"] else "false", shown)
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
    parts, flags = {}, []
    for piece in text.split(","):
        if "=" in piece:
            key, _, value = piece.partition("=")
            parts[key] = value
        else:
            flags.append(piece)
    if set(parts) != {"type", "source", "target"} or parts["type"] != "volume" or flags not in ([], ["readonly"]):
        refuse("mount %r (volume mounts only: type,source,target, and the bare flag readonly)" % text)
    if parts["source"] not in state["volumes"]:
        fail("docker: Error response from daemon: no such volume: " + parts["source"], 125)
    return {"name": parts["source"], "dest": parts["target"], "readonly": bool(flags)}


TOKEN_DIR = "/var/lib/raphael-tokens"
PLACEHOLDER = "invented placeholder text, never printed\n"


def token_paths(env):
    """The seven files the real owner app reads at startup: every *_TOKEN_FILE named in its settings."""
    return sorted(v for k, v in env.items() if k.endswith("_TOKEN_FILE"))


def volume_file(state, mounts, path):
    """(volume files, name) for a path under a mounted volume, or (None, None)."""
    folder, _, name = path.rpartition("/")
    for m in mounts:
        if m["dest"].rstrip("/") == folder:
            return state["volumes"][m["name"]]["files"], name
    return None, None


def owner_app_can_start(state, c):
    """The real owner app exits at startup when a token file is missing or empty (it never reads them here)."""
    for path in token_paths(c["env"]):
        files, name = volume_file(state, c["mounts"], path)
        if files is None or not files.get(name):
            return False
    return True


def after_start(state, c):
    """What a started container does at once: the owner app without all token files exits and stays down."""
    c["running"], c["status"], c["polls"] = True, "starting", state["polls"]
    c["gen"] = c.get("gen", 0) + 1
    if c["network"].startswith("container:"):
        _, owner = find_container(state, c["network"][len("container:"):])
        c["joined_gen"] = owner["gen"]
    if c["image_ref"].startswith("raphael-journeys-app:") and not owner_app_can_start(state, c):
        c["running"], c["status"] = False, "exited"


VALUE_FLAGS = {"--name", "--label", "--restart", "--network", "--network-alias", "--mount", "--env-file",
               "--health-cmd", "--health-interval", "--health-timeout", "--health-retries",
               "--health-start-period", "--user", "--entrypoint"}
BOOL_FLAGS = {"--detach", "--rm", "-i"}
# Docker refuses all of these together with --network container:X; this script uses none of them anywhere.
CLASH_VALUE = {"--hostname", "-h", "--dns", "--add-host", "--link", "--expose", "-p", "--publish"}
CLASH_BOOL = {"-P", "--publish-all"}


def run_helper(state, opts, command):
    """A throw-away helper container: no network, one volume, no name."""
    if set(opts) - {"--rm", "-i", "--network", "--user", "--entrypoint", "--mount", "--label", "--env-file"}:
        refuse("helper run flags")
    if opts.get("--network") != ["none"] or opts.get("--user") != ["0"] or opts.get("--entrypoint") != ["sh"]:
        refuse("helper run needs --network none --user 0 --entrypoint sh")
    if len(opts.get("--mount", [])) != 1 or len(command) != 2 or command[0] != "-c":
        refuse("helper run shape")
    mount = parse_mount(state, opts["--mount"][0])
    files = state["volumes"][mount["name"]]["files"]
    script = command[1]
    if mount["readonly"]:
        # The token check: it tests each named file with test -s and prints nothing; it cannot write.
        if opts.get("--label") != [LABEL_KEY + "=stack_up"] or opts.get("-i") or mount["dest"] != TOKEN_DIR:
            refuse("the read-only helper needs the label, no standard input and the token folder")
        env = read_env_file(opts.get("--env-file", [None])[0])
        tests = re.findall(r'test -s "\$([A-Z0-9_]+)"', script)
        if not tests or re.sub(r'test -s "\$[A-Z0-9_]+"|&&|true|\s', "", script):
            refuse("the token check may only test named files with test -s: %r" % script)
        for name in tests:
            path = env.get(name, "")
            folder, _, base = path.rpartition("/")
            if folder != mount["dest"] or not files.get(base):
                raise Done(1)
        return ""
    if opts.get("--label"):
        refuse("a writing helper is not made by this script")
    if "find /volume -mindepth 1 -delete" in script:
        files.clear()
    if opts.get("-i"):
        sys.stdin.buffer.read()
    return ""


def read_env_file(path_text):
    env = {}
    if path_text is None:
        return env
    path = Path(path_text)
    if not path.is_file():
        fail("docker: open %s: no such file or directory" % path, 125)
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        key, eq, value = line.partition("=")
        if not eq or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            refuse("env file line without NAME=value (a bare name would pass a host value through)")
        env[key] = value
    return env


def d_run(state, a, create=False):
    opts = {}
    i = 0
    while i < len(a) and a[i].startswith("-"):
        f = a[i]
        if f in BOOL_FLAGS or f in CLASH_BOOL:
            opts.setdefault(f, []).append(True)
            i += 1
        elif (f in VALUE_FLAGS or f in CLASH_VALUE) and i + 1 < len(a):
            opts.setdefault(f, []).append(a[i + 1])
            i += 2
        else:
            refuse("run flag %s (not expected)" % f)
    if i >= len(a):
        refuse("run without an image")
    image_ref, command = a[i], a[i + 1:]
    _, img = find_image(state, image_ref)
    if img is None:
        fail("docker: Unable to find image '%s' locally" % image_ref, 125)
    if opts.get("--rm"):
        if create:
            refuse("create --rm")
        return run_helper(state, opts, command)
    for f in ("--name", "--label", "--restart", "--network"):
        if len(opts.get(f, [])) != 1:
            refuse("run needs exactly one %s" % f)
    if bool(opts.get("--detach")) == create:
        refuse("run needs --detach; create must not have it")
    if opts["--label"] != [LABEL_KEY + "=stack_up"]:
        refuse("run label %r" % opts["--label"])
    if opts["--restart"] != ["unless-stopped"]:
        refuse("restart policy %r" % opts["--restart"])
    if set(opts) & {"--user", "--entrypoint"}:
        refuse("--user and --entrypoint are not expected")
    name, net = opts["--name"][0], opts["--network"][0]
    aliases = opts.get("--network-alias", [])
    clash = sorted((CLASH_VALUE | CLASH_BOOL) & set(opts))
    if net.startswith("container:") and (aliases or clash):
        refuse("conflicting options with --network container: %s" % ", ".join(clash + (["--network-alias"] if aliases else [])))
    if clash:
        refuse("no port may be published and none of %s is expected" % ", ".join(clash))
    if any(c["name"] == name for c in state["containers"].values()):
        fail('docker: Error response from daemon: Conflict. The container name "/%s" is already in use' % name, 125)
    if net.startswith("container:"):
        target = net[len("container:"):]
        if not HEX64.fullmatch(target):
            refuse("--network container: must name the container by its id")
        _, t = find_container(state, target)
        if t is None:
            fail("docker: Error response from daemon: No such container: " + target, 125)
        if t["labels"].get(LABEL_KEY) != "stack_up":
            refuse("joining the namespace of a container the script did not make")
        if not create and not t["running"]:
            fail("docker: Error response from daemon: cannot join network of a non running container: " + target, 125)
    elif net not in state["networks"]:
        fail("docker: Error response from daemon: network %s not found" % net, 125)
    env = read_env_file(opts.get("--env-file", [None])[0])
    mounts = [parse_mount(state, m) for m in opts.get("--mount", [])]
    health = img["health"]
    if "--health-cmd" in opts:
        health = ["CMD-SHELL", opts["--health-cmd"][0]]
    cid = new_id(state, "container")
    state["containers"][cid] = {
        "id": cid, "name": name, "image_ref": image_ref, "image_id": img["id"], "labels": {LABEL_KEY: "stack_up"},
        "env": env, "mounts": mounts, "network": net, "aliases": aliases, "ports": [], "restart": "unless-stopped",
        "health": health, "health_opts": {k: v[0] for k, v in opts.items() if k.startswith("--health-")},
        "cmd": command, "running": False, "status": "created", "polls": state["polls"], "gen": 0,
    }
    if not create:
        after_start(state, state["containers"][cid])
    if not create and image_ref.startswith("raphael-journeys-proxy:"):
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
            after_start(state, c)
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
        if "--out" not in cmd:
            refuse("token issue without --out")
        files, name = volume_file(state, c["mounts"], cmd[cmd.index("--out") + 1])
        if files is None:
            fail("hermes: cannot write the token file")
        files[name] = PLACEHOLDER
        return ""
    if cmd[:2] == ["sh", "-c"] and len(cmd) >= 3:
        script = cmd[2]
        if 'printf "%s %s\\n" "$w" "$n"' in script:
            return "default API_SERVER_KEY\nraphael-planner API_SERVER_KEY\n"
        if 'printf "entry %s\\n"' in script:
            return "entry raphael-planner\n"
        if 'exec "$@" -' in script and "profiles_to_serve" in stdin:
            return "served default\nserved raphael-planner\n"
        if "head -c 32 /dev/urandom" in script:
            files, name = volume_file(state, c["mounts"], cmd[5])
            if files is not None:
                files[name] = PLACEHOLDER
            return ""
        if 'cat >> "$1"' in script:
            return ""
    refuse("exec program %r" % (cmd[:3],))


def run_docker(argv):
    state = load()
    if not argv:
        refuse("no subcommand")
    sub, rest = argv[0], argv[1:]
    table = {"info": d_info, "inspect": d_inspect, "network": d_network, "volume": d_volume, "build": d_build,
             "run": d_run, "cp": d_cp, "rm": d_rm, "exec": d_exec}
    if sub == "create":
        return d_run(state, rest, create=True)
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

    def docker(self, *args: str) -> subprocess.CompletedProcess:
        """One call of the stand-in docker, as a person at a shell would make it."""
        return subprocess.run(
            [str(self.bin / "docker"), *args], env=self.env(), cwd=self.root, capture_output=True, text=True, timeout=60
        )

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


PLACEHOLDER = "invented placeholder text, never printed\n"
TOKEN_ENV_NAMES = sorted(name for name in SETTINGS if name.endswith("_TOKEN_FILE"))
TOKEN_NAMES = sorted(SETTINGS[name].rsplit("/", 1)[1] for name in TOKEN_ENV_NAMES)
FIRST_LINE = "stack_up: the first reset starts the owner app and the gateway"
ANSWERS_LINE = "stack_up: the stack answers"
RESET_LINE = "reset_stack: the stack is reset and every container answers"
GATEWAY_COMMAND = ["gateway", "run"]
DASHBOARD_COMMAND = ["dashboard", "--host", "0.0.0.0", "--port", "9119", "--no-open"]
IMAGE_REF = re.compile(r"raphael-journeys-(proxy|app|browser|platform):[A-Za-z0-9][A-Za-z0-9._-]*")


def seed_tokens(host: Host, missing: tuple[str, ...] = (), empty: tuple[str, ...] = ()) -> None:
    """A token volume that already holds the seven files (invented placeholder text); some may be missing or empty."""
    state = host.state()
    files = {name: PLACEHOLDER for name in TOKEN_NAMES if name not in missing}
    files.update({name: "" for name in empty})
    state["volumes"][TOKENS_V] = {"files": files}
    host.save(state)


@pytest.fixture
def host(tmp_path: Path) -> Host:
    return Host(tmp_path)


@pytest.fixture
def stack(host: Host) -> Host:
    """A converged stack that started from a token volume that already holds the files."""
    seed_tokens(host)
    result = host.converge()
    assert result.returncode == 0, result.stdout + result.stderr
    assert output_lines(result)[-1] == ANSWERS_LINE
    return host


def proxy_ca_pem(state: dict) -> str:
    return state["volumes"][PROXY_V]["files"]["ca.pem"]


def ca_part(state: dict) -> str:
    return hashlib.sha256(proxy_ca_pem(state).encode()).hexdigest()[:12]


def container_ids(host: Host) -> dict[str, str]:
    return {c["name"]: cid for cid, c in host.state()["containers"].items()}


def create_names(docker_calls: list[list[str]]) -> list[str]:
    return [argv[argv.index("--name") + 1] for argv in docker_calls if argv[0] == "create"]


def check_calls(docker_calls: list[list[str]]) -> list[list[str]]:
    """The read-only token check is the only throw-away helper container the converge runs."""
    return [argv for argv in docker_calls if argv[0] == "run" and "--rm" in argv]


def command_of(docker_calls: list[list[str]], name: str) -> list[str]:
    """The words after the image in the last recorded run or create call that made container NAME."""
    calls = [
        argv
        for argv in docker_calls
        if argv[0] in ("run", "create") and "--name" in argv and argv[argv.index("--name") + 1] == name
    ]
    assert calls, name
    argv = calls[-1]
    index = next(i for i, word in enumerate(argv) if IMAGE_REF.fullmatch(word))
    return argv[index + 1 :]


def namespaces_stale(state: dict) -> list[str]:
    """Running containers whose shared namespace is gone: the owner stopped, was removed, or restarted after them."""
    stale = []
    for c in state["containers"].values():
        if c["running"] and c["network"].startswith("container:"):
            owner = state["containers"].get(c["network"][len("container:") :])
            if owner is None or not owner["running"] or owner["gen"] != c["joined_gen"]:
                stale.append(c["name"])
    return stale


def check_topology(host: Host) -> None:
    """Shape of the five containers: rules 1 to 3 and decisions 1 to 7, true on first creation and when ready."""
    state = host.state()
    containers = host.containers()
    docker = host.docker_calls()
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
        if argv[0] in ("run", "create") and "--rm" not in argv:
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

    # Rule 3: the gateway joins the owner app's namespace (by id), not the reverse; the app carries the alias
    # workspace, the proxy's upstream; the gateway has no alias, hostname or port of its own.
    app_id = container_ids(host)[APP_C]
    assert containers[GATE_C]["network"] == f"container:{app_id}"
    assert containers[GATE_C]["aliases"] == []
    assert containers[APP_C]["network"] == NETWORK
    assert containers[APP_C]["aliases"] == ["workspace"]
    assert containers[DASH_C]["aliases"] == ["dashboard"]
    assert sorted(containers[PROXY_C]["aliases"]) == ["synthetic-team.cloudflareaccess.com", "workspace.synthetic.test"]
    for name in (PROXY_C, DASH_C, APP_C, BROWSER_C):
        assert containers[name]["network"] == NETWORK
    assert namespaces_stale(state) == []

    # Rule 2 and rule 1: the commands, as recorded in the create calls.
    assert command_of(docker, GATE_C) == GATEWAY_COMMAND
    assert containers[GATE_C]["cmd"] == GATEWAY_COMMAND
    assert command_of(docker, DASH_C) == DASHBOARD_COMMAND
    assert containers[DASH_C]["cmd"] == DASHBOARD_COMMAND
    assert containers[DASH_C]["health"][0] == "CMD-SHELL"
    assert "127.0.0.1:9119/api/health" in containers[DASH_C]["health"][1]
    assert "127.0.0.1:8642/health" in containers[GATE_C]["health"][1]


def assert_token_check(docker: list[list[str]]) -> None:
    """Rule 4 (c): one labelled, read-only helper that tests each named file with test -s and prints nothing."""
    helpers = check_calls(docker)
    assert len(helpers) == 1
    argv = helpers[0]
    assert argv[argv.index("--label") + 1] == LABEL
    assert argv[argv.index("--network") + 1] == "none"
    assert f"type=volume,source={TOKENS_V},target=/var/lib/raphael-tokens,readonly" in argv
    assert "-i" not in argv
    script = argv[-1]
    assert sorted(re.findall(r'test -s "\$([A-Z0-9_]+)"', script)) == TOKEN_ENV_NAMES
    assert not {"cat", "head", "od", "printf", "echo", "tee", ">", ">>"} & set(script.replace('"', " ").split())
    assert not [word for word in argv if "/var/lib/raphael-tokens/" in word]


def test_1_empty_host_with_token_files_creates_network_volumes_and_five_containers_in_order(host: Host) -> None:
    """Empty host, token volume already holds the files: the order, the shape of each container, rules 1 to 4."""
    seed_tokens(host)
    result = host.converge()
    assert result.returncode == 0, result.stdout + result.stderr
    assert output_lines(result)[-1] == ANSWERS_LINE

    docker = host.docker_calls()
    made = changing(docker)
    assert made[0] == ["network", "create", "--driver", "bridge", NETWORK]
    assert made[1:3] == [["volume", "create", name] for name in (HOME_V, PROXY_V)]
    assert [argv[0] for argv in made] == (
        ["network", "volume", "volume", "run", "cp", "build", "build", "run", "run", "run", "run", "run"]
    )
    assert run_names(docker) == [PROXY_C, DASH_C, APP_C, GATE_C, BROWSER_C]
    assert not [argv for argv in docker if argv[0] in ("create", "start", "restart", "stop")]
    check_topology(host)

    # The one token check comes after the dashboard and before the owner app and the gateway are started.
    assert_token_check(docker)
    helper = check_calls(docker)[0]
    app_call = next(a for a in docker if a[:1] == ["run"] and "--name" in a and APP_C in a)
    gate_call = next(a for a in docker if a[:1] == ["run"] and "--name" in a and GATE_C in a)
    assert docker.index(helper) < docker.index(app_call) < docker.index(gate_call)

    containers = host.containers()
    for name, container in containers.items():
        assert container["running"] and container["status"] == "healthy", name
    assert host.state()["volumes"][TOKENS_V]["files"] == {n: PLACEHOLDER for n in TOKEN_NAMES}
    for name in (DASH_C, GATE_C):
        image_env = host.state()["images"][containers[name]["image_ref"]]["env"]
        assert "HERMES_DASHBOARD" not in {**image_env, **containers[name]["env"]}, name


def test_1b_first_creation_creates_the_owner_app_and_gateway_without_starting_or_waiting_for_them(host: Host) -> None:
    """Empty host, empty token volume: the owner app and the gateway are created only; the fixed line; exit 0."""
    result = host.converge()
    assert result.returncode == 0, result.stdout + result.stderr
    assert output_lines(result)[-1] == FIRST_LINE
    assert ANSWERS_LINE not in output_lines(result)

    docker = host.docker_calls()
    made = changing(docker)
    assert made[1:4] == [["volume", "create", name] for name in STACK_VOLUMES]
    assert [argv[0] for argv in made] == (
        ["network", "volume", "volume", "volume", "run", "cp", "build", "build", "run", "run", "create", "create", "run"]
    )
    assert run_names(docker) == [PROXY_C, DASH_C, BROWSER_C]
    assert create_names(docker) == [APP_C, GATE_C]
    assert not [argv for argv in docker if argv[0] in ("start", "restart", "stop")]
    check_topology(host)
    assert_token_check(docker)

    ids = container_ids(host)
    containers = host.containers()
    for name in (APP_C, GATE_C):
        assert containers[name]["running"] is False and containers[name]["status"] == "created", name
    waited = {argv[5] for argv in docker if argv[0] == "inspect" and argv[4] == FORMATS["state"]}
    assert waited == {ids[PROXY_C], ids[DASH_C], ids[BROWSER_C]}
    for name in (PROXY_C, DASH_C, BROWSER_C):
        assert containers[name]["running"] and containers[name]["status"] == "healthy", name
    assert host.state()["volumes"][TOKENS_V]["files"] == {}


def test_2_reset_after_converge_exits_0_and_all_five_containers_are_running_and_healthy(stack: Host) -> None:
    """The unchanged reset_stack.sh on the converged stack: exit 0, every container healthy, tokens issued again."""
    mark = stack.mark()
    result = stack.run("reset_stack.sh")
    text = result.stdout + result.stderr
    assert result.returncode == 0, text
    assert output_lines(result)[-1] == RESET_LINE

    state = stack.state()
    containers = containers_by_name(state)
    assert sorted(containers) == sorted(STACK_CONTAINERS)
    for name, container in containers.items():
        assert container["running"] and container["status"] == "healthy", name
    assert namespaces_stale(state) == []
    assert all(state["volumes"][TOKENS_V]["files"].get(name) for name in TOKEN_NAMES)

    order = [(argv[0], argv[1]) for argv in stack.docker_calls(mark) if argv[0] in ("stop", "start", "restart")]
    assert order.index(("stop", GATE_C)) < order.index(("restart", APP_C)) < order.index(("start", GATE_C))


def test_3_isolation_check_names_on_template_passes_on_the_created_stack(stack: Host) -> None:
    """isolation_check.sh server2: the names-on-template check passes against the same state."""
    result = stack.run("isolation_check.sh", "server2")
    assert "PASS names-on-template" in result.stdout.splitlines(), result.stdout + result.stderr


def test_4_second_converge_with_the_same_tags_makes_no_create_remove_volume_or_token_check_call(stack: Host) -> None:
    """A second converge with all five running and the same tags changes nothing and checks nothing."""
    before = {c["name"]: cid for cid, c in stack.state()["containers"].items()}
    mark = stack.mark()
    result = stack.converge()
    assert result.returncode == 0, result.stdout + result.stderr
    calls = stack.docker_calls(mark)
    assert {argv[0] for argv in calls} <= {"info", "inspect", "cp"}, calls
    assert {c["name"]: cid for cid, c in stack.state()["containers"].items()} == before
    assert output_lines(result)[-1] == ANSWERS_LINE


def test_4b_first_creation_run_again_while_files_are_missing_changes_nothing_and_ends_the_same(host: Host) -> None:
    """Files still missing: nothing is created, started or written; the same last line; the check only reads."""
    assert host.converge().returncode == 0
    before = host.state()
    mark = host.mark()
    result = host.converge()
    assert result.returncode == 0, result.stdout + result.stderr
    assert output_lines(result)[-1] == FIRST_LINE
    calls = host.docker_calls(mark)
    assert {argv[0] for argv in calls} <= {"info", "inspect", "cp", "run"}, calls
    assert len(check_calls(calls)) == 1 and len([argv for argv in calls if argv[0] == "run"]) == 1
    assert not [argv for argv in calls if argv[0] in ("create", "start", "restart", "stop", "rm", "build")]
    assert host.state() == before


def test_4c_when_the_files_exist_the_stopped_owner_app_is_started_first_then_the_gateway(host: Host) -> None:
    """After a first creation the files appear: start the app, then the gateway, and wait for all five."""
    assert host.converge().returncode == 0
    ids = container_ids(host)
    seed_tokens(host)
    mark = host.mark()
    result = host.converge()
    assert result.returncode == 0, result.stdout + result.stderr
    assert output_lines(result)[-1] == ANSWERS_LINE
    calls = host.docker_calls(mark)
    assert [argv for argv in calls if argv[0] in ("start", "restart", "stop")] == [["start", ids[APP_C]], ["start", ids[GATE_C]]]
    assert not [argv for argv in calls if argv[0] in ("create", "rm", "volume", "network", "build")]
    assert [argv for argv in calls if argv[0] == "run"] == check_calls(calls) and len(check_calls(calls)) == 1
    state = host.state()
    for name, container in containers_by_name(state).items():
        assert container["running"] and container["status"] == "healthy", name
    assert container_ids(host) == ids
    assert namespaces_stale(state) == []
    assert state["volumes"][TOKENS_V]["files"] == {n: PLACEHOLDER for n in TOKEN_NAMES}


@pytest.mark.parametrize("index", range(7))
def test_4d_a_missing_or_zero_size_token_file_is_a_first_creation(tmp_path: Path, index: int) -> None:
    """Any one of the seven files missing (even index) or of zero size (odd index): nothing starts, same last line."""
    host = Host(tmp_path)
    name = TOKEN_NAMES[index]
    seed_tokens(host, missing=(name,) if index % 2 == 0 else (), empty=() if index % 2 == 0 else (name,))
    files = host.state()["volumes"][TOKENS_V]["files"]
    result = host.converge()
    assert result.returncode == 0, result.stdout + result.stderr
    assert output_lines(result)[-1] == FIRST_LINE
    docker = host.docker_calls()
    assert create_names(docker) == [APP_C, GATE_C]
    assert not [argv for argv in docker if argv[0] in ("start", "restart", "stop")]
    assert not [c for c in host.containers().values() if c["name"] in (APP_C, GATE_C) and c["running"]]
    assert host.state()["volumes"][TOKENS_V]["files"] == files


def test_5_new_owner_app_tag_recreates_the_owner_app_the_gateway_and_the_browser(stack: Host) -> None:
    """A new owner app tag: three removals by id, three creations, no volume call, no proxy volume in a remove."""
    before = container_ids(stack)
    mark = stack.mark()
    result = stack.converge(app=NEW_REVISION)
    assert result.returncode == 0, result.stdout + result.stderr
    docker = stack.docker_calls(mark)

    removed = [argv for argv in docker if argv[0] == "rm"]
    assert sorted(argv[2] for argv in removed) == sorted([before[APP_C], before[GATE_C], before[BROWSER_C]])
    assert all(argv[:2] == ["rm", "--force"] and len(argv) == 3 for argv in removed)
    assert run_names(docker) == [APP_C, GATE_C, BROWSER_C]
    assert len(check_calls(docker)) == 1
    assert not [argv for argv in docker if argv[0] in ("volume", "network")]

    after = container_ids(stack)
    for name in (PROXY_C, DASH_C):
        assert after[name] == before[name], name
    for name in (APP_C, GATE_C, BROWSER_C):
        assert after[name] != before[name], name
    state = stack.state()
    ca = ca_part(state)
    assert containers_by_name(state)[APP_C]["image_ref"] == f"raphael-journeys-app:{NEW_REVISION}-{ca}"
    assert containers_by_name(state)[GATE_C]["network"] == f"container:{after[APP_C]}"
    assert command_of(docker, GATE_C) == GATEWAY_COMMAND
    assert namespaces_stale(state) == []
    assert sorted(state["volumes"]) == sorted(STACK_VOLUMES)

    for argv in stack.docker_calls():
        assert argv[0] not in ("system", "image", "rmi", "prune")
        if argv[0] == "volume":
            assert argv[1] == "create"
        if argv[0] == "rm":
            assert not {"-v", "--volumes", PROXY_V, PROXY_C} & set(argv)
            assert re.fullmatch(r"[0-9a-f]{64}", argv[-1])


def test_5b_gateway_is_recreated_when_its_network_mode_does_not_name_the_current_owner_app(stack: Host) -> None:
    """Recovery after a failed converge: a gateway that joined an owner app that is gone is made again."""
    state = stack.state()
    containers_by_name(state)[GATE_C]["network"] = "container:" + "0" * 64
    stack.save(state)
    before = container_ids(stack)
    mark = stack.mark()
    result = stack.converge()
    assert result.returncode == 0, result.stdout + result.stderr
    docker = stack.docker_calls(mark)
    assert [argv[2] for argv in docker if argv[0] == "rm"] == [before[GATE_C]]
    assert run_names(docker) == [GATE_C]
    after = container_ids(stack)
    assert after[APP_C] == before[APP_C] and after[GATE_C] != before[GATE_C]
    assert containers_by_name(stack.state())[GATE_C]["network"] == f"container:{after[APP_C]}"
    assert namespaces_stale(stack.state()) == []


def test_5c_a_running_gateway_beside_a_stopped_owner_app_is_made_again_after_the_app_starts(stack: Host) -> None:
    """The namespace the gateway joined is gone once the app stopped: start the app, then make the gateway again."""
    state = stack.state()
    containers_by_name(state)[APP_C]["running"] = False
    stack.save(state)
    before = container_ids(stack)
    mark = stack.mark()
    result = stack.converge()
    assert result.returncode == 0, result.stdout + result.stderr
    assert output_lines(result)[-1] == ANSWERS_LINE
    docker = stack.docker_calls(mark)
    assert [argv for argv in docker if argv[0] in ("rm", "start", "restart", "stop")] == [
        ["rm", "--force", before[GATE_C]],
        ["start", before[APP_C]],
    ]
    assert run_names(docker) == [GATE_C]
    gate_run = next(i for i, argv in enumerate(docker) if argv[0] == "run" and "--name" in argv)
    assert docker.index(["start", before[APP_C]]) < gate_run
    after = container_ids(stack)
    assert after[APP_C] == before[APP_C] and after[GATE_C] != before[GATE_C]
    state = stack.state()
    assert namespaces_stale(state) == []
    for name, container in containers_by_name(state).items():
        assert container["running"] and container["status"] == "healthy", name
    assert containers_by_name(state)[GATE_C]["network"] == f"container:{after[APP_C]}"


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
        if argv[0] in ("run", "create"):
            mounts = [argv[i + 1] for i, word in enumerate(argv) if word == "--mount"]
            names_proxy_volume = any(f"source={PROXY_V}," in m for m in mounts)
            named = argv[argv.index("--name") + 1] if "--name" in argv else None
            assert names_proxy_volume == (named == PROXY_C)


def test_8_no_output_line_or_command_line_carries_an_environment_value(host: Host, tmp_path: Path) -> None:
    """Values reach docker only through --env-file; neither the output nor any command line shows one."""
    first = sub_host(tmp_path, "first")
    seed_tokens(host)
    results = [first.converge(), host.converge(), host.converge(app=NEW_REVISION)]
    results.append(host.run("stack_up.sh", "converge", PROXY_TAG, "bad tag", PLATFORM, REVISION))
    results.append(host.run("stack_up.sh", "versions"))
    state = host.state()
    app_env = containers_by_name(state)[APP_C]["env"]
    assert app_env == SETTINGS
    values = {value for value in SETTINGS.values() if value} | {"/state", "/opt/data"}
    for result in results:
        for line in output_lines(result):
            assert not [v for v in values if v in line], line
            assert PLACEHOLDER.strip() not in line
    for each in (first, host):
        for argv in each.docker_calls():
            for word in argv:
                if word == "--mount" or word.startswith("type=volume,"):
                    continue
                assert not [v for v in values - {"/state", "/opt/data"} if v in word and len(v) > 4], word
                assert PLACEHOLDER.strip() not in word
            assert "-e" not in argv and "--env" not in argv
            if argv[0] in ("run", "create") and "--name" in argv and argv[argv.index("--name") + 1] != BROWSER_C:
                assert "--env-file" in argv
            if argv[0] == "run" and "--rm" in argv:
                assert "--env-file" in argv and "--name" not in argv


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


def test_extra_script_is_executable_and_wrong_usage_exits_2_before_any_docker_call(host: Host) -> None:
    mode = (JOURNEYS / "stack_up.sh").stat().st_mode & 0o777
    assert mode & 0o755 == 0o755 and not mode & 0o002  # rwxr-xr-x at least; a checkout may add group write (775)
    for args in ([], ["build"], ["build", "nothing"], ["build", "app"], ["converge", PROXY_TAG], ["versions", "x"]):
        result = host.run("stack_up.sh", *args)
        assert result.returncode == 2, args
        assert result.stderr.startswith("usage: stack_up.sh"), args
    assert host.docker_calls() == []


def test_extra_new_platform_commit_recreates_the_dashboard_and_the_gateway_and_keeps_the_owner_app(stack: Host) -> None:
    """A new platform commit: dashboard and gateway are removed by id and made again; the new gateway joins the app."""
    state = stack.state()
    seed_image(state, f"raphael-journeys-platform:{NEW_PLATFORM}", env={"PATH": "/usr/bin", "HERMES_HOME": "/opt/data"})
    stack.save(state)
    before = container_ids(stack)
    mark = stack.mark()
    result = stack.converge(platform=NEW_PLATFORM)
    assert result.returncode == 0, result.stdout + result.stderr
    docker = stack.docker_calls(mark)
    removed = [argv[2] for argv in docker if argv[0] == "rm"]
    assert sorted(removed) == sorted([before[DASH_C], before[GATE_C]])
    assert run_names(docker) == [DASH_C, GATE_C]
    assert len(check_calls(docker)) == 1
    after = container_ids(stack)
    for name in (PROXY_C, APP_C, BROWSER_C):
        assert after[name] == before[name], name
    containers = containers_by_name(stack.state())
    assert containers[GATE_C]["network"] == f"container:{after[APP_C]}"
    assert containers[GATE_C]["image_ref"] == containers[DASH_C]["image_ref"] == f"raphael-journeys-platform:{NEW_PLATFORM}"
    assert command_of(docker, GATE_C) == GATEWAY_COMMAND and command_of(docker, DASH_C) == DASHBOARD_COMMAND
    assert namespaces_stale(stack.state()) == []
    assert not [argv for argv in docker if argv[0] in ("volume", "network")]


def test_extra_container_that_never_answers_fails_in_one_budget_and_converge_resumes(tmp_path: Path) -> None:
    """No answer: exit 1 after at most 90 polls of 2 seconds; fixing the cause and converging again makes no change."""
    host = Host(tmp_path, never_healthy=[BROWSER_C])
    seed_tokens(host)
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


# --- round 2: what the stand-in itself refuses and models (finding 2 and finding 3)


@pytest.mark.parametrize("damage", ["missing", "zero_size"])
def test_9_a_started_owner_app_without_all_seven_token_files_exits_at_once_and_is_never_healthy(
    stack: Host, damage: str
) -> None:
    """The stand-in models the real owner app: it exits at once and stays down when any named file is missing or empty."""
    app_id = container_ids(stack)[APP_C]
    for name in TOKEN_NAMES:
        state = stack.state()
        if damage == "missing":
            del state["volumes"][TOKENS_V]["files"][name]
        else:
            state["volumes"][TOKENS_V]["files"][name] = ""
        stack.save(state)
        assert stack.docker("restart", app_id).returncode == 0
        shown = stack.docker("inspect", "--type", "container", "--format", FORMATS["state"], app_id).stdout.strip()
        assert shown == "false none", name
        assert containers_by_name(stack.state())[APP_C]["running"] is False
        state = stack.state()
        state["volumes"][TOKENS_V]["files"][name] = PLACEHOLDER
        stack.save(state)
    assert stack.docker("restart", app_id).returncode == 0
    for _ in range(3):
        shown = stack.docker("inspect", "--type", "container", "--format", FORMATS["state"], app_id).stdout.strip()
    assert shown == "true healthy"


def test_9b_converge_does_not_wait_for_an_owner_app_that_exits_at_once(tmp_path: Path) -> None:
    """With the files gone after a first creation the stand-in app would exit; converge never starts it."""
    host = Host(tmp_path)
    seed_tokens(host, empty=(TOKEN_NAMES[0],))
    result = host.converge()
    assert result.returncode == 0 and output_lines(result)[-1] == FIRST_LINE
    assert [c["name"] for c in host.containers().values() if c["running"]] == [PROXY_C, DASH_C, BROWSER_C]


def test_10_the_stand_in_refuses_what_the_engine_refuses_for_a_shared_namespace(stack: Host) -> None:
    """Finding 3: no start while the container to join is not running; no alias, hostname, port, dns, add-host,
    expose or link together with --network container:."""
    ids = container_ids(stack)
    state_before = stack.state()
    platform_ref = f"raphael-journeys-platform:{PLATFORM}"
    base = ["run", "--detach", "--label", LABEL, "--restart", "unless-stopped", "--name", "stand-in-probe"]
    joined = ["--network", f"container:{ids[APP_C]}"]
    for extra in (
        ["--network-alias", "workspace"],
        ["--hostname", "probe"],
        ["-p", "8080:80"],
        ["--publish", "8080:80"],
        ["--dns", "192.0.2.1"],
        ["--add-host", "probe:192.0.2.2"],
        ["--expose", "80"],
        ["--link", "somewhere"],
    ):
        result = stack.docker(*base, *joined, *extra, platform_ref, *GATEWAY_COMMAND)
        assert result.returncode != 0 and "conflicting options" in result.stderr, extra
    assert stack.state() == state_before

    assert stack.docker("stop", GATE_C).returncode == 0
    assert stack.docker("stop", APP_C).returncode == 0
    refused = stack.docker("start", GATE_C)
    assert refused.returncode != 0 and "cannot join network of a non running container" in refused.stderr
    refused = stack.docker(*base, *joined, platform_ref, *GATEWAY_COMMAND)
    assert refused.returncode != 0 and "cannot join network of a non running container" in refused.stderr
    assert "stand-in-probe" not in stack.containers()
    assert stack.docker("start", APP_C).returncode == 0
    assert stack.docker("start", GATE_C).returncode == 0


# --- round 2: the gateway container runs one gateway (rule 2, finding 4)


@pytest.mark.parametrize("created", ["started", "created_only"])
def test_11_the_recorded_create_call_pins_the_plain_gateway_run_and_no_dashboard_variable(tmp_path: Path, created: str) -> None:
    """Rule 2: the command after the image is exactly gateway run, without --no-supervise, in both creation paths;
    HERMES_DASHBOARD is in no env name or value, and the dashboard container keeps its arguments and order."""
    host = Host(tmp_path)
    if created == "started":
        seed_tokens(host)
    result = host.converge()
    assert result.returncode == 0, result.stdout + result.stderr
    docker = host.docker_calls()
    gateway_call = next(a for a in docker if a[0] in ("run", "create") and "--name" in a and GATE_C in a)
    assert gateway_call[0] == ("run" if created == "started" else "create")
    assert command_of(docker, GATE_C) == ["gateway", "run"]
    assert "--no-supervise" not in gateway_call
    assert not [word for word in gateway_call if "HERMES_DASHBOARD" in word]
    assert command_of(docker, DASH_C) == ["dashboard", "--host", "0.0.0.0", "--port", "9119", "--no-open"]
    for argv in docker:
        assert not [word for word in argv if "HERMES_DASHBOARD" in word]
    state = host.state()
    for container in state["containers"].values():
        image_env = state["images"][container["image_ref"]]["env"]
        merged = {**image_env, **container["env"]}
        assert not [k for k in merged if "HERMES_DASHBOARD" in k]
        assert not [v for v in merged.values() if "HERMES_DASHBOARD" in v]
        gateways = [c for c in state["containers"].values() if c["cmd"][:2] == ["gateway", "run"]]
        assert [c["name"] for c in gateways] == [GATE_C]


# --- round 2: the gateway container runs one gateway, shown with the real boot decisions (rule 2, finding 4)
#
# These tests call the real functions with the container command that the converge recorded for the gateway
# container: the legacy-migration check and reconcile_profile_gateways(dry_run=True) of container_boot (the
# cont-init step of the image) and gateway_command -> _cmd_run -> _maybe_redirect_run_to_s6_supervision of
# gateway (the command process). The s6 service manager is a recording fake, run_gateway, os.execvp and
# _block_until_terminated are recorders, subprocess.Popen raises, and every file lives under tmp_path. No
# Python source text is read and no process is started.

ROUND1_CMD = ["gateway", "run", "--no-supervise"]

# argv of the container's main program as ``container_boot._read_container_argv`` sees it.
# ``docker/entrypoint-dispatch.sh`` execs ``/init <wrapper> "$@"`` (no ``hermes`` token); the
# wrapper-with-``hermes`` shape is the one ``_strip_container_argv_prefix`` also documents.
_INIT = "/init"
_WRAPPER = "/opt/hermes/docker/main-wrapper.sh"
ARGV_PREFIXES = [
    pytest.param((_INIT, _WRAPPER, "hermes"), id="wrapper-hermes"),
    pytest.param((_INIT, _WRAPPER), id="entrypoint-dispatch"),
]

# On-disk ``$HERMES_HOME/gateway_state.json`` per home kind. ``_read_desired_state`` returns
# ``desired_state`` when present, else ``gateway_state`` (``draining`` / ``degraded`` read as
# ``running``). ``empty`` has no state file at all.
_TS = 1234567890
HOME_STATE: dict[str, dict | None] = {
    "empty": None,
    "running": {"gateway_state": "running", "timestamp": _TS},
    "degraded": {"gateway_state": "degraded", "timestamp": _TS},
    "draining": {"gateway_state": "draining", "timestamp": _TS},
    # Shape ``S6ServiceManager.start`` writes via ``_write_gateway_desired_state``.
    "desired_running": {"desired_state": "running", "updated_at": _TS},
    # What an operator stop leaves behind (``S6ServiceManager.stop`` + the gateway's own persist).
    "stopped": {"gateway_state": "stopped", "desired_state": "stopped", "updated_at": _TS},
}
PERSISTED_RUNNING = ("running", "degraded", "draining", "desired_running")

# (profile, prior_state, action) of the default slot the boot reconcile decides, per home kind.
_STARTED = ("default", "running", "started")
EXPECTED_DEFAULT_SLOT = {
    "empty": _STARTED,  # the legacy migration seeds ``running`` for a bare ``gateway run``
    "running": _STARTED,
    "degraded": _STARTED,
    "draining": _STARTED,
    "desired_running": _STARTED,
    "stopped": ("default", "stopped", "registered"),  # registered DOWN; the redirect starts it
}
HOME_PARAMS = [pytest.param(kind, id=kind) for kind in HOME_STATE]


class _RecordingS6:
    """Stand-in for ``S6ServiceManager``: records lifecycle calls and starts nothing."""

    kind = "s6"

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def start(self, name: str) -> None:
        self.calls.append(("start", name))
        # Same on-disk effect as the real ``S6ServiceManager.start`` (writes desired_state
        # ``running`` into the profile's gateway_state.json); a pure file write under tmp.
        from hermes_cli.service_manager import _write_gateway_desired_state

        _write_gateway_desired_state(name, "running")

    def stop(self, name: str) -> None:
        self.calls.append(("stop", name))

    def restart(self, name: str) -> None:
        self.calls.append(("restart", name))


class _NoProcess(subprocess.Popen):
    """Any attempt to start a real process fails the test."""

    def __init__(self, *args, **kwargs):
        raise AssertionError(f"boot decisions must not start a process: {args!r}")


def _noop(_args) -> None:
    return None


def _parse_gateway_args(cmd) -> argparse.Namespace:
    """Parse the container command with the real ``gateway`` argparse tree (``--no-supervise``)."""
    from hermes_cli.subcommands.gateway import build_gateway_parser

    parser = argparse.ArgumentParser(prog="hermes")
    subparsers = parser.add_subparsers(dest="command")
    build_gateway_parser(
        subparsers, cmd_gateway=_noop, cmd_proxy=_noop, cmd_gateway_enroll=_noop,
    )
    return parser.parse_args(list(cmd))


def boot_decisions(
    cmd, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, home_kind: str,
    *, argv_prefix: tuple[str, ...] = (_INIT, _WRAPPER, "hermes"),
) -> dict:
    """Run the real boot decisions for container command ``cmd`` over a ``home_kind`` home.

    Order, as in the real boot: legacy migration + reconcile (cont-init, root), then the command
    process (``hermes <cmd>``). Returns plain data:

      supervised         sorted distinct s6 services started (reconcile 'started' + redirect start)
      foreground_runs    ``run_gateway`` calls made by the command process
      execvp             argv lists passed to ``os.execvp`` by the command process
      redirected         the redirect decision returned True (and was consulted exactly once)
      reconcile_actions  [(profile, prior_state, action), ...] from the dry-run reconcile
      owners             ``supervised`` plus one ``"foreground"`` per foreground gateway
      migration          return of the legacy migration check (``"running"`` or None)
      service_calls      every call the redirect made on the service manager
      heartbeat_fallbacks   times the in-process ``_block_until_terminated`` fallback ran
      desired_state_after   ``_read_desired_state(home)`` once the boot decisions are done
    """
    # Importing ``hermes_cli.gateway`` scaffolds ``$HERMES_HOME`` (SOUL.md, sessions/, ...) and
    # ``tests/conftest.py`` sandboxes HOME/HERMES_HOME before any import for that reason. Do the
    # same here so the helper also leaves the real home alone when it is run on its own.
    monkeypatch.setenv("HOME", str(tmp_path / "import-home"))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "import-home" / ".hermes"))
    from hermes_cli import container_boot
    from hermes_cli import gateway as gw

    home = tmp_path / "hermes-home"
    home.mkdir()
    state = HOME_STATE[home_kind]
    if state is not None:
        (home / "gateway_state.json").write_text(json.dumps(state) + "\n", encoding="utf-8")
    scandir = tmp_path / "run-service"  # dry run: reconcile must not write s6 slots here

    monkeypatch.setenv("HERMES_HOME", str(home))
    for name in ("HERMES_S6_SUPERVISED_CHILD", "HERMES_GATEWAY_NO_SUPERVISE",
                 "GATEWAY_MULTIPLEX_PROFILES"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(subprocess, "Popen", _NoProcess)

    # Seams of test_container_boot.py / test_gateway_s6_dispatch.py.
    monkeypatch.setattr("hermes_cli.container_boot._read_container_argv", lambda: ())
    rec = _RecordingS6()
    monkeypatch.setattr("hermes_cli.service_manager.detect_service_manager", lambda: "s6")
    monkeypatch.setattr("hermes_cli.service_manager.get_service_manager", lambda: rec)
    monkeypatch.setattr("hermes_cli.gateway._profile_suffix", lambda: "")
    execvp_calls: list[list[str]] = []
    monkeypatch.setattr(
        "hermes_cli.gateway.os.execvp", lambda _file, args: execvp_calls.append(list(args)),
    )
    block_calls: list[bool] = []
    monkeypatch.setattr(
        "hermes_cli.gateway._block_until_terminated", lambda: block_calls.append(True),
    )
    foreground: list[tuple] = []
    monkeypatch.setattr(
        "hermes_cli.gateway.run_gateway", lambda *a, **kw: foreground.append((a, kw)),
    )
    redirects: list[bool] = []
    real_redirect = gw._maybe_redirect_run_to_s6_supervision

    def _spy_redirect(args) -> bool:
        redirects.append(real_redirect(args))
        return redirects[-1]

    monkeypatch.setattr("hermes_cli.gateway._maybe_redirect_run_to_s6_supervision", _spy_redirect)

    # Boot, step 1 (cont-init): legacy migration, then reconcile (dry run).
    container_argv = [*argv_prefix, *cmd]
    migration = container_boot._maybe_migrate_legacy_gateway_run_state(
        home, container_argv=container_argv, dry_run=False,
    )
    actions = container_boot.reconcile_profile_gateways(
        hermes_home=home, scandir=scandir, dry_run=True, container_argv=container_argv,
    )
    # Boot, step 2: the command process, ``hermes gateway run ...`` (-> _cmd_run -> redirect).
    gw.gateway_command(_parse_gateway_args(cmd))

    started_by_reconcile = {f"gateway-{a.profile}" for a in actions if a.action == "started"}
    started_by_redirect = {name for verb, name in rec.calls if verb == "start"}
    supervised = sorted(started_by_reconcile | started_by_redirect)
    return {
        "supervised": supervised,
        "foreground_runs": len(foreground),
        "execvp": execvp_calls,
        "redirected": redirects == [True],
        "reconcile_actions": [(a.profile, a.prior_state, a.action) for a in actions],
        "owners": [*supervised, *(["foreground"] * len(foreground))],
        "migration": migration,
        "service_calls": list(rec.calls),
        "heartbeat_fallbacks": len(block_calls),
        "desired_state_after": container_boot._read_desired_state(home),
    }


@pytest.fixture(scope="module")
def recorded_gateway_command(tmp_path_factory: pytest.TempPathFactory) -> list[str]:
    """The words after the image in the recorded create call of the gateway container."""
    host = Host(tmp_path_factory.mktemp("recorded_gateway"))
    seed_tokens(host)
    result = host.converge()
    assert result.returncode == 0, result.stdout + result.stderr
    return command_of(host.docker_calls(), GATE_C)


@pytest.mark.parametrize("argv_prefix", ARGV_PREFIXES)
@pytest.mark.parametrize("home_kind", HOME_PARAMS)
def test_12_the_recorded_gateway_command_launches_exactly_one_gateway_owner_for_every_home(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    recorded_gateway_command: list[str],
    home_kind: str,
    argv_prefix: tuple[str, ...],
) -> None:
    """Empty, persisted running (running, degraded, draining, desired_state running) and explicitly stopped homes:
    one supervised gateway-default, and the command process is only the heartbeat."""
    result = boot_decisions(recorded_gateway_command, tmp_path, monkeypatch, home_kind, argv_prefix=argv_prefix)

    # What the boot decided for this home (the legacy migration only seeds the empty home).
    assert result["migration"] == ("running" if home_kind == "empty" else None)
    assert result["reconcile_actions"] == [EXPECTED_DEFAULT_SLOT[home_kind]]
    # (1) exactly one supervised service, started once by the redirect (idempotent with the reconcile).
    assert result["supervised"] == ["gateway-default"]
    assert result["service_calls"] == [("start", "gateway-default")]
    # (2) the command process runs no foreground gateway.
    assert result["foreground_runs"] == 0
    # (3) it hands over and is only the "sleep infinity" heartbeat (not the in-process fallback).
    assert result["redirected"] is True
    assert result["execvp"] == [["sleep", "infinity"]]
    assert result["heartbeat_fallbacks"] == 0
    assert result["owners"] == ["gateway-default"]


@pytest.mark.parametrize("home_kind", [pytest.param(kind, id=kind) for kind in PERSISTED_RUNNING])
def test_12b_control_the_round_1_command_is_two_gateway_owners_on_a_persisted_running_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, home_kind: str
) -> None:
    """Shows that the assertions above can fail: with --no-supervise the redirect opts out, so the boot starts
    gateway-default from the persisted state and the command process runs a second, foreground gateway."""
    result = boot_decisions(ROUND1_CMD, tmp_path, monkeypatch, home_kind)
    assert result["redirected"] is False
    assert result["execvp"] == []
    assert result["service_calls"] == []
    assert result["supervised"] == ["gateway-default"]
    assert result["foreground_runs"] == 1
    assert result["owners"] == ["gateway-default", "foreground"]
