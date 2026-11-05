"""Shared helpers, fakes, fixtures and constants of the journeys stack_up tests.

The three test_journeys_stack_up*.py files split the stack_up.sh tests by run time and import what they use
from here.

The script runs against stand-ins for docker, sleep, hostname and df that sit first on PATH. The stand-in
docker records every call in calls.jsonl, keeps a fake Server 2 in state.json, and refuses (status 98) any
subcommand or flag the script is not expected to use. It also models what the real engine and the real owner
app do that matter here: it refuses to start a container that joins the namespace of a container that is not
running, refuses a --network container: container that also asks for an alias, hostname, published port, dns,
add-host, expose or link, keeps the files of each volume with their sizes, gives each container a
State.StartedAt from a strictly increasing fake clock at every start and restart (printed the way Docker prints
it: trailing zeros of the fraction cut, no fraction when it is zero, the zero time while never started; a test
can overwrite it with an exact string), and makes a started owner app exit
at once when one of the seven token files of stack-app.settings is missing or empty. DOCKER_HOST points at a
path that does not exist, so a real docker could not answer. No real Docker call is made anywhere. The stand-in
can also answer one identity read by script (scripted.json next to state.json), print the daemon's own wording of a
missing container (STUB_NOT_FOUND=daemon) and, while STUB_VERIFY is set (every run of stack_up.sh), refuse with
status 98 a start, stop, restart, rm, cp or --network container:ID of an id that no identity read of that run showed
as owned (verified.txt), which makes owner rule 2 of round 4 an invariant of every converge test. It answers the
identity read the way Docker's own template does: the id, one space and the word owned when the stored value of the
label raphael-journeys.made-by is exactly stack_up (Python equality on the raw value), otherwise the word foreign,
and never a label value. The tests read recorded calls, output lines and the fake host state; they never read the
script's text, and the token files hold invented placeholder text that is never printed.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

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
    "ident": '{{.Id}} {{if eq (index .Config.Labels "raphael-journeys.made-by") "stack_up"}}owned{{else}}foreign{{end}}',
    "config_image": "{{.Config.Image}}",
    "network_mode": "{{.HostConfig.NetworkMode}}",
    "started_at": "{{.State.StartedAt}}",
    "name": "{{.Name}}",
    "id": "{{.Id}}",
    "security": "{{range .SecurityOptions}}{{println .}}{{end}}",
    "root_dir": "{{.DockerRootDir}}",
}

STUB = r'''
import os
import sys

# Each stand-in call is a fresh interpreter, and the stdlib json, re, pathlib and hashlib imports cost more than the
# rest of the call. The common path (reads and writes of the state and the calls log) uses the C coder of json
# directly and the other modules are imported only by the subcommands that need them.
try:
    import _json

    class _Context:
        strict = True
        object_hook = object_pairs_hook = None
        parse_float, parse_int = float, int
        memo = {}

        @staticmethod
        def parse_constant(name):
            return {"-Infinity": float("-inf"), "Infinity": float("inf"), "NaN": float("nan")}[name]

    _scan = _json.make_scanner(_Context)
    _encode = _json.make_encoder(None, None, _json.encode_basestring_ascii, None, ": ", ", ", False, False, True)

    class json:
        @staticmethod
        def loads(text):
            text = text.strip()
            value, end = _scan(text, 0)
            if end != len(text):
                raise ValueError("Extra data")
            return value

        @staticmethod
        def dumps(value):
            return "".join(_encode(value, 0))
except Exception:
    import json


def read_text(path):
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def write_text(path, text):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
STATE_FILE = os.path.join(ROOT, "state.json")
CALLS = os.path.join(ROOT, "calls.jsonl")
LABEL_KEY = "raphael-journeys.made-by"
TOOL = ""


class HEX64:
    @staticmethod
    def fullmatch(text):
        return len(text) == 64 and all(ch in "0123456789abcdef" for ch in text)


class Done(Exception):
    def __init__(self, rc, err=""):
        self.rc, self.err = rc, err


def refuse(why):
    raise Done(98, "stand-in %s: refused: %s\n" % (TOOL, why))


def fail(message, rc=1):
    raise Done(rc, message + "\n")


VERIFIED = os.path.join(ROOT, "verified.txt")
SCRIPTED = os.path.join(ROOT, "scripted.json")


def verified_ids():
    return read_text(VERIFIED).split() if os.path.exists(VERIFIED) else []


def note_verified(cid):
    with open(VERIFIED, "a", encoding="utf-8") as handle:
        handle.write(cid + "\n")


def require_verified(cid):
    """While STUB_VERIFY is set, an id reaches a changing call only after a read of this run showed the label."""
    if os.environ.get("STUB_VERIFY") == "1" and cid not in verified_ids():
        refuse("id %s was not read with the label stack_up in this run" % cid)


def changed_since(start):
    for line in read_text(CALLS).splitlines()[start:]:
        call = json.loads(line)
        if call["tool"] == "docker" and call["argv"][:1] not in (["info"], ["inspect"]):
            return True
    return False


def scripted_ident(name):
    """The scripted entry that fires for this identity read of container NAME (see Host.script_read), or None."""
    if not os.path.exists(SCRIPTED):
        return None
    entries = json.loads(read_text(SCRIPTED))
    for entry in entries:
        if entry["name"] != name or entry["fired"]:
            continue
        if entry["after_change"] and not changed_since(entry["since"]):
            continue
        hit = None
        if entry["seen"] < entry["skip"]:
            entry["seen"] += 1
        else:
            entry["fired"] += 1
            hit = entry
        write_text(SCRIPTED, json.dumps(entries))
        return hit
    return None


def load():
    return json.loads(read_text(STATE_FILE))


def save(state):
    tmp = os.path.join(ROOT, "state.tmp.%d" % os.getpid())
    write_text(tmp, json.dumps(state))
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
    import hashlib

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
        if fmt == F["ident"]:
            hit = scripted_ident(c["name"] if c else ref)
            if hit and hit["status"] != 0:
                raise Done(hit["status"], hit["stderr"])
            if hit:
                sys.stderr.write(hit["stderr"])
                if hit["stdout"] is not None:
                    return hit["stdout"]
        if c is None:
            if os.environ.get("STUB_NOT_FOUND") == "daemon":
                fail("Error response from daemon: No such container: " + ref)
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
            owned = c["labels"].get(LABEL_KEY) == "stack_up"
            out = "%s %s" % (cid, "owned" if owned else "foreign")
            if owned:
                note_verified(cid)
        elif fmt == F["config_image"]:
            out = c["image_ref"]
        elif fmt == F["network_mode"]:
            out = c["network"]
        elif fmt == F["started_at"]:
            out = c.get("started_at", ZERO_TIME)
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
    import hashlib
    import re
    from pathlib import Path

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


ZERO_TIME = "0001-01-01T00:00:00Z"


def fake_time(tick):
    """Docker's State.StartedAt shape: RFC 3339 UTC, the fraction with its trailing zeros cut, none when zero.
    The clock runs in steps of an eighth of a second from 18:00:00Z, so whole seconds (no fraction) and short and
    long fractions all occur, and a later value can sort lower as text (18:00:01.125Z against 18:00:01Z)."""
    seconds, part = divmod(tick, 8)
    hours, rest = divmod(18 * 3600 + seconds, 3600)
    minutes, secs = divmod(rest, 60)
    fraction = ("%09d" % (part * 125000000)).rstrip("0")
    return "2026-10-28T%02d:%02d:%02d%sZ" % (hours, minutes, secs, "." + fraction if fraction else "")


def after_start(state, c):
    """What a started container does at once: the owner app without all token files exits and stays down."""
    c["running"], c["status"], c["polls"] = True, "starting", state["polls"]
    c["gen"] = c.get("gen", 0) + 1
    state["clock"] = state.get("clock", 0) + 1
    c["started_at"] = fake_time(state["clock"])
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
    import re

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
    if state.get("fail_next_writing_helper"):
        # One transient engine failure, injected by a test: the helper changes nothing and the flag is spent.
        state["fail_next_writing_helper"] = False
        save(state)
        fail("docker: Error response from daemon: transient failure (injected by the test)", 125)
    if "find /volume -mindepth 1 -delete" in script:
        files.clear()
    if opts.get("-i"):
        sys.stdin.buffer.read()
    return ""


def read_env_file(path_text):
    import re
    from pathlib import Path

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
        require_verified(target)
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
        "started_at": ZERO_TIME,
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
    from pathlib import Path

    if len(a) != 2 or ":" not in a[0]:
        refuse("cp shape %r (copy out of a container only)" % a)
    cid, _, path = a[0].partition(":")
    if not HEX64.fullmatch(cid):
        refuse("cp must name the container by its id")
    require_verified(cid)
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
    require_verified(a[1])
    if a[1] not in state["containers"]:
        fail("Error response from daemon: No such container: " + a[1])
    del state["containers"][a[1]]
    save(state)
    return a[1] + "\n"


def d_lifecycle(state, a, sub):
    if len(a) != 1:
        refuse("%s shape %r" % (sub, a))
    require_verified(a[0])
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
    # The reset's board store, Project and board calls as the platform's account: each is matched whole, word by word,
    # in its place.
    anchor, dashboard = "raphael-workspace", "raphael-journeys-dashboard"
    if (a[:i], a[i]) == (["--user", "10000:10000"], dashboard) and cmd == [
        "hermes", "kanban", "boards", "create", anchor,
    ]:
        return ""
    if (a[:i], a[i]) == (["--user", "10000:10000"], dashboard) and cmd == [
        "hermes", "project", "create", "Raphael Workspace", "--slug", anchor, "--board", anchor,
    ]:
        return ""
    # BOARD_SHELL and BOARD_PROGRAM are the reset's texts, put in front of this source by the test as F is.
    if (a[:i], a[i]) == (["-i", "--user", "10000:10000"], dashboard) and (
        cmd == ["sh", "-c", BOARD_SHELL, "sh"] and stdin == BOARD_PROGRAM
    ):
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
        "clock": 0,
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
            wrapper.write_text(
                f'#!/bin/sh\nexec "{sys.executable}" -S -I -c '
                f"'import sys; sys.path.insert(0, sys.argv.pop(1)); import stub; stub.main()' "
                f'"${{0%/*}}" {tool} "$@"\n',
                encoding="utf-8",
            )
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
        if script == "stack_up.sh":
            # Each run starts with no verified id: the stand-in then refuses an id that no read of this run showed
            # with the label (owner rule 2 of round 4).
            extra_env["STUB_VERIFY"] = "1"
            (self.root / "verified.txt").write_text("", encoding="utf-8")
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

    def docker(self, *args: str, **extra_env: str) -> subprocess.CompletedProcess:
        """One call of the stand-in docker, as a person at a shell would make it."""
        return subprocess.run(
            [str(self.bin / "docker"), *args],
            env=self.env(**extra_env),
            cwd=self.root,
            capture_output=True,
            text=True,
            timeout=60,
        )

    def script_read(
        self, name: str, status: int = 1, stdout: str | None = "", stderr: str = "", skip: int = 0, after_change: bool = False
    ) -> None:
        """Script one identity read of container NAME: the first skip reads are let through, the next one fires once.
        Status 0 with a string answers that stdout; status 0 with stdout None passes the real answer and only writes
        stderr (a warning); a non-zero status fails with stderr. With after_change only reads that follow the first
        changing docker call logged after this call count."""
        path = self.root / "scripted.json"
        entries = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
        entries.append(
            {"name": name, "skip": skip, "after_change": after_change, "status": status, "stdout": stdout,
             "stderr": stderr, "seen": 0, "fired": 0, "since": len(self.calls())}
        )
        path.write_text(json.dumps(entries), encoding="utf-8")

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


@pytest.fixture(scope="module")
def converged_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The converged host, built once: a stack that started from a token volume that already holds the files."""
    host = Host(tmp_path_factory.mktemp("converged"))
    seed_tokens(host)
    result = host.converge()
    assert result.returncode == 0, result.stdout + result.stderr
    assert output_lines(result)[-1] == ANSWERS_LINE
    return host.root


def private_copy(converged_root: Path, root: Path) -> Host:
    """A copy of the converged host at ROOT with an empty calls log."""
    shutil.copytree(converged_root, root, symlinks=True)
    converge_log = [json.loads(line) for line in (root / "calls.jsonl").read_text(encoding="utf-8").splitlines()]
    for name in ("state.json", "scripted.json", "verified.txt"):
        path = root / name
        if path.exists():
            path.write_text(path.read_text(encoding="utf-8").replace(str(converged_root), str(root)), encoding="utf-8")
    (root / "calls.jsonl").write_text("", encoding="utf-8")
    copy = Host.__new__(Host)
    copy.root, copy.scripts, copy.bin = root, root / "tree" / "scripts" / "journeys", root / "bin"
    copy.converged_docker = [c["argv"] for c in converge_log if c["tool"] == "docker"]
    return copy


@pytest.fixture
def stack(converged_root: Path, tmp_path: Path) -> Host:
    """A private copy of the converged host with an empty calls log."""
    return private_copy(converged_root, tmp_path / "stack")


@pytest.fixture(scope="module")
def shared_stack(converged_root: Path, tmp_path_factory: pytest.TempPathFactory) -> Host:
    """One copy of the converged host for all the cases of test_2b to test_2e: each makes one docker exec call that is
    taken or refused with no change of the host (only the calls log grows), and none of them reads the calls log."""
    return private_copy(converged_root, tmp_path_factory.mktemp("shared") / "stack")


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


# The reset's board store, Project and board calls; the stand-in takes each only as the whole argument list the reset
# passes, and the board write only with the reset's whole shell text and its whole program on standard input.
ANCHOR = "raphael-workspace"
AS_PLATFORM = ["--user", "10000:10000"]
# BOARD_SHELL and BOARD_PROGRAM are a deliberate copy of the text that reset_stack.sh sends for the board metadata
# write: platform_python as the shell text, and anchor_program for ANCHOR plus the one newline of the here-string as
# the input. The test does not read the reset, so a change of the reset turns the test red.
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
# The stand-in compares with these very texts: they go in front of its source as F does.
STUB = f"BOARD_SHELL = {BOARD_SHELL!r}\nBOARD_PROGRAM = {BOARD_PROGRAM!r}\n" + STUB


def lifecycle_calls(docker: list[list[str]]) -> list[list[str]]:
    """Every call that creates, removes, starts or stops one of the five containers (not the --rm token check)."""
    return [argv for argv in docker if argv[0] in ("run", "create", "start", "stop", "restart", "rm") and "--rm" not in argv]


def is_health_inspect(argv: list[str]) -> bool:
    """An inspect of State.Health: what the health wait polls (the census reads the same format before any change)."""
    return argv[0] == "inspect" and "State.Health" in argv[4]


def assert_health_wait_after_lifecycle(docker: list[list[str]]) -> None:
    """From the first to the last lifecycle call no health is inspected; the health wait comes after the last one.
    Inspects of the id, the network mode and State.StartedAt are allowed between the lifecycle calls."""
    lifecycle = [i for i, argv in enumerate(docker) if lifecycle_calls([argv])]
    assert lifecycle
    health = [i for i, argv in enumerate(docker) if is_health_inspect(argv) and i > lifecycle[0]]
    assert health and min(health) > lifecycle[-1]


def assert_all_running_and_healthy(host: Host) -> None:
    state = host.state()
    containers = containers_by_name(state)
    assert sorted(containers) == sorted(STACK_CONTAINERS)
    for name, container in containers.items():
        assert container["running"] and container["status"] == "healthy", name
    assert namespaces_stale(state) == []


def assert_no_volume_removal(docker: list[list[str]]) -> None:
    for argv in docker:
        assert argv[0] not in ("system", "image", "rmi", "prune")
        if argv[0] == "volume":
            assert argv[1] == "create"
        if argv[0] == "rm":
            assert not {"-v", "--volumes", PROXY_V, PROXY_C} & set(argv)


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


def sub_host(parent: Path, name: str, **overrides) -> Host:
    root = parent / name
    root.mkdir()
    return Host(root, **overrides)


@pytest.fixture(scope="module")
def recorded_gateway_command(tmp_path_factory: pytest.TempPathFactory) -> list[str]:
    """The words after the image in the recorded create call of the gateway container."""
    host = Host(tmp_path_factory.mktemp("recorded_gateway"))
    seed_tokens(host)
    result = host.converge()
    assert result.returncode == 0, result.stdout + result.stderr
    return command_of(host.docker_calls(), GATE_C)
