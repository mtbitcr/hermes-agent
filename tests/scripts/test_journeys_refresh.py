"""Tests for the journeys refresh script (card 3: refresh_stack.sh on Server 1).

refresh_stack.sh runs for real, from a temporary copy of the journeys folder inside a temporary
"checkout", against stand-ins that sit alone on PATH: ssh, docker, git, timeout, date and sleep (plus
symlinks to bash and dirname). Every stand-in logs its call to one file, in order, and answers from
answers.json. Only the fake ssh (Server 2) keeps state: state.json is read and rewritten inside an
exclusive flock and is published by rename, so no process ever sees half of it; the git and docker of
Server 1 never touch it. The fake ssh records [destination, command, stdin size and sha256], answers by
the kind of command and may advance the clock; it runs the commands that touch files for real in a temporary
Server 2 folder (the marker read, the scripts stage and publish, the platform archive stage, publish and
removal, and the page check stage and publish against a fake docker that maps each container to a folder).
Its mv and rm are wrappers that fail one named rename or removal while a control file exists in the sandbox.
A producer may exit non-zero after a complete stream (late_rc) or after a stream cut between entries (cut),
and the tests compare what is on the fake Server 2 before and after a failed run: nothing may be published
until every side of a transfer has reported. The clock is a number in a file that is published by rename:
date +%s prints it, sleep N adds N and returns at once, timeout logs its seconds and runs the wrapped command.
No test reaches a real host, docker or ssh. Runs of the script are shared between tests that look at the same
scenario.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import stat
import subprocess
import sys
import threading
from collections import Counter
from pathlib import Path

import pytest

pytestmark = pytest.mark.linux_only

REPO = Path(__file__).resolve().parents[2]
JOURNEYS = REPO / "scripts" / "journeys"
SCRIPT = JOURNEYS / "refresh_stack.sh"
TEMPLATE = JOURNEYS / "stack.env.template"

SECRET = "sk-ant-oat01-SENTINEL-secret-value-0123456789"
RUNNER = "runner@server2.invalid"
COMMIT = "c0ffee" * 6 + "c0ff"
OTHER_COMMIT = "d1ce" * 10
REVISION = "5e1f" * 10
IMAGE_ID = "sha256:" + "ab12" * 16
PLATFORM_ID = "sha256:" + "cd34" * 16
PROXY_TAG = "0123456789ab"
CA = "cacacacacaca"
APP_TAG = f"{REVISION}-{CA}"
PLATFORM_REF = f"raphael-journeys-platform:{COMMIT}"
APP_BASE_TAG = f"raphael-journeys-app-base:{REVISION}"
CLOCK_START = 1_700_000_000
BOX = 1500
OPT = "/opt/raphael-page-check"
BROWSER_PAGE_CHECK = "/opt/raphael-journeys/tools/page-check"

SUCCESS = "refresh_stack: the stack is at the released versions\n"
REASONS = {
    "usage": "no argument is accepted",
    "config": "a setting or the template is missing or is not plain",
    "commit": "the platform commit could not be read",
    "gateway": "the gateway code version could not be read or is absent",
    "differ": "the platform commit and the gateway code version differ",
    "owner": "the owner app could not be read",
    "revision": "the owner app revision is not 40 lowercase hex",
    "ship-scripts": "the scripts folder could not be shipped to Server 2",
    "marker-ask": "Server 2 could not be asked which scripts folder it holds",
    "image-ask": "Server 2 could not be asked about an image",
    "app-image": "the owner app image could not be loaded or tagged on Server 2",
    "platform-build": "the platform build could not be shipped or started",
    "box": "the 25-minute box ran out; a platform build that is running keeps running",
    "proxy-tag": "the proxy tag was not reported",
    "converge": "the stack could not be converged",
    "page-check": "the page check could not be copied into the browser container",
    "prune": "old images could not be listed or removed",
}
REPOSITORIES = [
    "raphael-journeys-proxy", "raphael-journeys-app-base", "raphael-journeys-app",
    "raphael-journeys-platform", "raphael-journeys-browser",
]

STAND_IN = r'''
import json, os, sys

top = os.path.dirname(os.path.abspath(__file__))
name = os.path.basename(sys.argv[1])
args = sys.argv[2:]
with open(os.path.join(top, "answers.json"), encoding="utf-8") as handle:
    A = json.load(handle)
SECRET = A["secret"]
log, clock, state_path = (os.path.join(top, n) for n in ("calls.jsonl", "clock", "state.json"))


def now():
    with open(clock, encoding="utf-8") as handle:
        return int(handle.read())


def sha(data):
    import hashlib
    return hashlib.sha256(data).hexdigest()


def publish(target, text):
    # temporary file in the same folder, then a rename: a reader sees the old file or the whole new one
    temp = "%s.%d.tmp" % (target, os.getpid())
    half = len(text) // 2
    with open(temp, "w", encoding="utf-8") as handle:
        handle.write(text[:half])
        handle.flush()
        pause()
        handle.write(text[half:])
    os.replace(temp, target)


def locked(path):
    import fcntl
    handle = open(path + ".lock", "a", encoding="utf-8")
    fcntl.flock(handle, fcntl.LOCK_EX)
    return handle


def pause():
    if os.environ.get("STAND_IN_PAUSE"):
        import time
        time.sleep(float(os.environ["STAND_IN_PAUSE"]))


def bump_clock(seconds):
    if seconds:
        with locked(clock):
            value = now() + seconds
            pause()
            publish(clock, str(value))


def record(label, **extra):
    with open(log, "a", encoding="utf-8") as handle:
        handle.write(json.dumps({"call": label, "argv": [name, *args], **extra}) + "\n")


def end(rc, out=None, data=None, err=None):
    # every stand-in is noisy with a value that must never come out of the script
    if data is not None:
        sys.stdout.buffer.write(data)
    elif out is not None:
        sys.stdout.write(out)
    else:
        sys.stdout.write("leak: %s\n" % SECRET)
    sys.stderr.write("leak: %s\n" % SECRET)
    if err is not None:
        sys.stderr.write(err)
    sys.stdout.flush()
    sys.exit(rc)


def make_tar(files, cut=None):
    # cut: the name of an entry; the archive then ends at the header boundary before it, without end-of-archive blocks
    import io, tarfile
    buf, seen, end_at = io.BytesIO(), set(), None
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for path in sorted(files):
            if cut and end_at is None and path.endswith("/" + cut):
                end_at = buf.tell()
            parts = path.split("/")
            for i in range(1, len(parts)):
                folder = "/".join(parts[:i])
                if folder not in seen:
                    seen.add(folder)
                    info = tarfile.TarInfo(folder)
                    info.type, info.mode = tarfile.DIRTYPE, 0o755
                    tar.addfile(info)
            blob = files[path].encode("utf-8")
            info = tarfile.TarInfo(path)
            info.size, info.mode = len(blob), (0o755 if path.endswith(".sh") else 0o644)
            tar.addfile(info, io.BytesIO(blob))
    return buf.getvalue() if end_at is None else buf.getvalue()[:end_at]


if name == "git":
    where, rest = (args[1], args[2:]) if args[:1] == ["-C"] else (os.getcwd(), args)
    if rest[:2] == ["rev-parse", "--show-toplevel"]:
        label, out = "git toplevel", A["toplevel"] + "\n"
    elif rest[:2] == ["rev-parse", "HEAD"]:
        label, out = "git head", A["head"] + "\n"
    elif rest[:1] == ["archive"]:
        label, out = ("git archive scripts" if "scripts/journeys" in rest else "git archive commit"), None
    else:
        record("git other", dir=where)
        end(98, out="")
    rc = A["rc"].get(label, 0)
    if out is not None or rc:
        record(label, dir=where)
        if out is not None:
            bump_clock(A["advance"].get(label, 0))
        end(rc, out="" if rc else out)
    data = make_tar(A["scripts_files"] if label.endswith("scripts") else A["commit_files"], A["cut"].get(label))
    record(label, dir=where, sha=sha(data))
    end(A["late_rc"].get(label, 0), data=data)

if name == "docker":
    if args[:1] == ["inspect"]:
        record("docker inspect")
        bump_clock(A["advance"].get("docker inspect", 0))
        end(A["rc"].get("docker inspect", A["inspect"]["rc"]), out=A["inspect"]["out"])
    if args[:1] == ["save"]:
        data = A["save_data"].encode("utf-8")
        rc = A["rc"].get("docker save", 0)
        record("docker save", sha=sha(data))
        end(rc or A["late_rc"].get("docker save", 0), data=b"" if rc else data)
    record("docker other")
    end(98, out="")
'''

STAND_IN += r'''
if name == "ssh":
    import shlex
    command = args[-1]
    data = b"" if sys.stdin.isatty() else sys.stdin.buffer.read()
    D, B = A["s2_dir"], A["browser"]
    ref, archive = "raphael-journeys-platform:" + A["head"], "%s.platform-%s.tar" % (D, A["head"])
    exact = {
        f"cd '{D}' && ./stack_up.sh versions": "versions",
        f"cd '{D}' && ./stack_up.sh build proxy": "build-proxy",
        f"docker exec '{B}' test -d '/opt/raphael-journeys/tools/page-check'": "page-check-present",
        "docker load": "load",
        f"if [ -e '{D}.new' ]; then rm -r '{D}.new'; fi": "rm-scripts-new",
        f"if [ -e '{D}.page-check.tar' ]; then rm '{D}.page-check.tar'; fi": "rm-staging-tar",
    }
    prefixes = [
        ("docker tag ", "tag"), ("cat > ", "ship-platform"),
        (f"cd '{D}' && ./stack_up.sh build platform ", "build-platform"),
        (f"cd '{D}' && ./stack_up.sh converge ", "converge"),
        ("docker image inspect ", "has-image"),
        ("docker image ls ", "list-tags"), ("docker image rm ", "remove-tag"),
        ("r=0; ", "rm-archive"),
        ("echo ", "publish-scripts"), ("mv ", "publish-platform"),
    ]
    REAL_KINDS = ("marker", "ship-scripts", "publish-scripts", "ship-platform", "publish-platform", "rm-archive",
                  "copy-page-check", "publish-page-check", "rm-scripts-new", "rm-page-check", "rm-staging-tar")
    kind = exact.get(command)
    if kind is None:
        if " tar -xf - -C " in command:
            kind = "ship-scripts"
        elif "docker cp " in command:
            kind = "copy-page-check"
        elif command.startswith("docker exec ") and 'for d in "$@"' in command:
            kind = "rm-page-check"
        elif ".refresh-commit" in command and " cat " in command and not command.startswith("echo "):
            kind = "marker"
        elif command.startswith("docker exec "):
            kind = "publish-page-check"
        else:
            kind = next((k for p, k in prefixes if command.startswith(p)), "other")
    try:
        words = shlex.split(command)
    except ValueError:
        words = []
    merged = words[-1:] == ["2>&1"]
    last = words[-1] if words else ""
    core = words[:-1] if merged else words
    first = words[3] if len(words) > 3 and not words[3].startswith("-") else ""
    arg = {"has-image": core[-1] if core else "", "tag": last, "remove-tag": last, "list-tags": first}.get(kind, "")
    label = (kind + " " + arg).strip()
    record(kind, remote=command, dest=args[0], nargs=len(args), stdin_size=len(data),
           stdin_sha=sha(data), secret_in_stdin=SECRET.encode("utf-8") in data, arg=arg)
    real, rc, out, err = False, 0, None, None
    with locked(state_path):
        if os.path.exists(state_path):
            with open(state_path, encoding="utf-8") as handle:
                state = json.load(handle)
        else:
            state = {"images": list(A["images"]), "untagged": list(A["untagged"]),
                     "tags": {k: list(v) for k, v in A["tags"].items()},
                     "built_at": None, "converged": False, "count": {}}
        n = state["count"][kind] = state["count"].get(kind, 0) + 1

        def pick(table, default=None):
            for key in (label, "%s#%d" % (kind, n), kind):
                if key in table:
                    return table[key]
            return default

        forced, advance = pick(A["rc"]), pick(A["advance"], 0)
        if kind == "list-tags":
            out = "".join(t + "\n" for t in state["tags"].get(arg, []))
        elif kind == "versions":
            out = A["out"].get("versions#%d" % n)
            if out is None:
                out = A["out"]["versions_after"] if state["converged"] else A["out"]["versions"]
        elif kind in ("build-proxy", "converge"):
            out = A["out"][kind]
        if kind in ("list-tags", "versions", "build-proxy", "converge"):
            rc = forced or 0
        elif forced is not None and not (forced == 0 and kind in REAL_KINDS):
            rc = forced
        elif kind in REAL_KINDS:
            real = True
        elif kind == "has-image":
            if (arg == ref and A["ready_after"] is not None and state["built_at"] is not None
                    and now() - state["built_at"] >= A["ready_after"] and ref not in state["images"]):
                state["images"].append(ref)
            # like docker image inspect: an image held by id is found whether or not it has a tag; a missing one
            # is status 1 with the error on stderr (and an empty line or [] on stdout)
            if arg in state["images"]:
                out = (A["platform_id"] if arg == ref else arg) + "\n"
            else:
                rc, out, err = 1, A["missing_stdout"], A["missing_error"].replace("%s", arg)
            garbled = pick(A["garbled"])
            if garbled is not None:
                out, err = garbled, None
            if merged and err is not None:
                out, err = out + err, None
        elif kind == "load":
            if A["app_image_id"] not in state["images"]:
                state["untagged"].append(A["app_image_id"])
            state["images"].append(A["app_image_id"])
        elif kind == "tag":
            known = words[2] in state["images"]
            if known:
                state["images"].append(words[3])
                state["untagged"] = [i for i in state["untagged"] if i != words[2]]
            rc = 0 if known else 1
        elif kind == "remove-tag":
            repo, tag = arg.split(":", 1)
            if tag in state["tags"].get(repo, []):
                state["tags"][repo].remove(tag)
        elif kind == "build-platform":
            rc = 0 if os.path.exists(last) else 1
        elif kind != "page-check-present":
            rc = 98
        if kind == "has-image" and forced is not None:
            out = pick(A["garbled"])
        if kind == "build-platform" and rc == 0:
            state["built_at"] = now() + advance
            with open(last, "rb") as handle:
                state["archive_sha"] = sha(handle.read())
            if A["eats_archive"]:
                os.remove(last)
        if kind == "converge" and rc == 0:
            state["converged"] = True
        pause()
        publish(state_path, json.dumps(state))
    if real:
        import subprocess
        env = {"PATH": os.path.join(top, "s2bin"), "HOME": top}
        done = subprocess.run(["/bin/sh", "-c", command], env=env, input=data, capture_output=True, cwd=top)
        rc, out = done.returncode, (done.stdout.decode("utf-8") if kind == "marker" else None)
    late = pick(A["late_rc"])
    if late is not None:
        rc = late
    bump_clock(advance)
    end(rc, out=out, err=err)

if name == "s2-docker":
    # Server 2 docker as far as the page check copy needs it: each container is a folder of s2fs
    fs = os.path.join(top, "s2fs")

    def host(container, path):
        return os.path.join(fs, container, path.lstrip("/"))

    op = {"exec": "exec", "cp": "cp-out" if args[2:3] == ["-"] else "cp-in"}.get(args[0], "other")
    record("s2docker " + op)
    if op == "exec":
        import subprocess
        if not os.path.isdir(os.path.join(fs, args[1])):
            sys.exit(1)
        mapped = [host(args[1], a) if a.startswith("/") else a for a in args[2:]]
        sys.exit(subprocess.run(mapped).returncode)
    if op == "cp-out":
        import tarfile
        container, src = args[1].split(":", 1)
        base = host(container, src)
        if not os.path.isdir(base):
            sys.exit(1)
        failing = A["rc"].get("s2docker cp-out", 0)
        found = []
        for folder, dirs, files in os.walk(base):
            dirs.sort()
            found += [os.path.relpath(os.path.join(folder, e), base) for e in sorted(dirs + files)]
        with tarfile.open(fileobj=sys.stdout.buffer, mode="w|") as tar:
            for rel in sorted(found):
                if failing and rel.endswith("helper.js"):
                    continue  # a valid archive that stops short of the whole folder
                tar.add(os.path.join(base, rel), arcname="./" + rel, recursive=False)
        sys.exit(failing or A["late_rc"].get("s2docker cp-out", 0))
    if op == "cp-in":
        import io, tarfile
        container, dst = args[2].split(":", 1)
        raw = sys.stdin.buffer.read()
        if A["rc"].get("s2docker cp-in") or not os.path.isdir(host(container, dst)):
            sys.exit(1)
        if raw:
            with tarfile.open(fileobj=io.BytesIO(raw), mode="r") as tar:
                tar.extractall(host(container, dst))
        sys.exit(0)
    sys.exit(98)
sys.exit("stand-in: unknown name " + name)
'''


def versions_text(proxy=PROXY_TAG, app=APP_TAG, dashboard=COMMIT, gateway=COMMIT, browser=APP_TAG) -> str:
    return f"proxy {proxy}\napp {app}\ndashboard {dashboard}\ngateway {gateway}\nbrowser {browser}\n"


RECONCILE = ["rm-scripts-new", "rm-archive", "rm-page-check", "rm-staging-tar"]
GATE = ["marker", "has-image", "has-image"]
FULL_RUN = [
    "marker", "has-image", "has-image", "ship-scripts", "publish-scripts", "load", "tag", "ship-platform",
    "publish-platform", "build-platform", "has-image", "build-proxy", "converge", "copy-page-check",
    "publish-page-check", *RECONCILE, "versions", *["list-tags"] * 5,
]
CURRENT_RUN = ["marker", "versions", "page-check-present", *RECONCILE, "versions", *["list-tags"] * 5]
CACHED_RUN = [
    "marker", "versions", "has-image", "has-image", "tag", "build-proxy", "converge", "copy-page-check",
    "publish-page-check", *RECONCILE, "versions", *["list-tags"] * 5,
]
SHIPPING = {"ship-scripts", "publish-scripts", "load", "ship-platform", "publish-platform", "build-platform"}
INSPECT_APP = f"docker image inspect --format '{{{{.Id}}}}' '{IMAGE_ID}' 2>&1"
INSPECT_PLATFORM = f"docker image inspect --format '{{{{.Id}}}}' '{PLATFORM_REF}' 2>&1"


def tokens(text: str) -> list[str]:
    found = [t.strip("'\"") for t in re.findall(r"""'[^']*'|"[^"]*"|[^\s;&|(){}]+""", text)]
    for word in list(found):
        if re.search(r"\s", word):
            found += tokens(word)
    return found


def forbidden(text: str) -> list[str]:
    """The words no call may carry: -f and --force as tokens, prune, volume, docker rm, container rm."""
    words = tokens(text)
    bad = [w for w in words if w in ("-f", "--force", "prune", "volume")]
    bad += [f"{a} {b}" for a, b in zip(words, words[1:]) if b == "rm" and a in ("docker", "container")]
    return bad


def values_in(text: str) -> list[str]:
    values = (COMMIT, OTHER_COMMIT, REVISION, IMAGE_ID, IMAGE_ID[7:], PROXY_TAG, CA, SECRET)
    return [value for value in values if value in text]


def expect_failure(done: subprocess.CompletedProcess[str], key: str) -> None:
    assert done.stderr == f"refresh_stack: {REASONS[key]}\n", done.stderr
    assert done.stdout == "", done.stdout
    assert done.returncode == 1
    assert not values_in(done.stdout + done.stderr)


def expect_success(done: subprocess.CompletedProcess[str]) -> None:
    assert done.stderr == "", done.stderr
    assert done.stdout == SUCCESS, done.stdout
    assert done.returncode == 0
    assert not values_in(done.stdout + done.stderr)


def tree(folder: Path) -> dict[str, str]:
    return {
        str(path.relative_to(folder)): path.read_text(encoding="utf-8")
        for path in sorted(folder.rglob("*")) if path.is_file()
    }


class Sandbox:
    """A temporary checkout with a copy of the journeys folder, the stand-ins and the answers they give."""

    def __init__(self, top: Path) -> None:
        self.top = top
        self.runs: list[tuple[int, int]] = []
        self.checkout = top / "checkout"
        self.scripts = self.checkout / "scripts" / "journeys"
        self.bin, self.s2bin, self.home = top / "bin", top / "s2bin", top / "hermes-home"
        self.s2dir = top / "s2" / "raphael-journeys"
        self.fs = top / "s2fs"
        for folder in (self.scripts, self.bin, self.s2bin, self.home, self.s2dir.parent):
            folder.mkdir(parents=True)
        (top / "stand_in.py").write_text(STAND_IN, encoding="utf-8")
        (top / "clock").write_text(str(CLOCK_START), encoding="utf-8")
        for name in ("ssh", "docker", "git"):
            self._wrapper(self.bin / name, name)
        self._wrapper(self.s2bin / "docker", "s2-docker")
        self._clock_tools()
        for name in ("bash", "dirname"):
            (self.bin / name).symlink_to(shutil.which(name))
        for name in ("sh", "tar", "mkdir", "cat"):
            (self.s2bin / name).symlink_to(shutil.which(name))
        self._failing_tool("mv", '"$1"', {
            "*page-check.new": "mv-fails", f'"{self.s2dir}"': "mv-fails-first", f'"{self.s2dir}.new"': "mv-fails-second",
        })
        self._failing_tool("rm", '"$@"', {
            "*page-check.old": "rm-fails", f'"{self.s2dir}.previous"': "rm-fails-previous",
            "*page-check.tar": "rm-fails-staging", "*platform-*.tar": "rm-fails-archive", "*.tar.part": "rm-fails-part",
            f'"{self.s2dir}.new"': "rm-fails-new",
        })
        shutil.copy2(TEMPLATE, self.scripts / "stack.env.template")
        self._names()
        self.answers = {
            "secret": SECRET, "s2_dir": str(self.s2dir), "browser": self.browser_c,
            "toplevel": str(self.checkout), "head": COMMIT, "app_image_id": IMAGE_ID,
            "inspect": {"rc": 0, "out": f"{REVISION} {IMAGE_ID}\n"}, "save_data": "SAVED-IMAGE-BYTES " * 40,
            "scripts_files": {
                "scripts/journeys/stack_up.sh": "#!/bin/sh\n# new stack_up\n",
                "scripts/journeys/stack.env.template": "[containers]\n",
                "scripts/journeys/images/proxy.Dockerfile": "FROM scratch\n",
            },
            "commit_files": {"Dockerfile": "FROM scratch\n", "app/main.py": "print(1)\n"},
            "images": [], "untagged": [], "tags": {}, "ready_after": 0, "rc": {}, "advance": {}, "late_rc": {}, "cut": {},
            "garbled": {}, "platform_id": PLATFORM_ID, "eats_archive": False,
            "missing_stdout": "\n", "missing_error": "Error response from daemon: No such image: %s\n",
            "out": {
                "versions": versions_text("none", "none", "none", "none", "none"),
                "versions_after": versions_text(),
                "build-proxy": f"stack_up: noise line\nstack_up: built proxy {PROXY_TAG}\n",
                "converge": "stack_up: making things\nstack_up: the stack answers\n",
            },
        }
        self.answer()
        self.seed_fs()
        self.write_state(COMMIT)

    def _wrapper(self, path: Path, name: str) -> None:
        path.write_text(f'#!/bin/sh\nexec "{sys.executable}" -I -S "{self.top / "stand_in.py"}" {name} "$@"\n')
        path.chmod(0o755)

    def _failing_tool(self, tool: str, looked_at: str, rules: dict[str, str]) -> None:
        """A wrapper of mv or rm for the fake Server 2: it exits 1 when a word it looks at matches a pattern and the
        control file of that pattern exists in the sandbox folder; every other call goes to the real tool."""
        lines = ["#!/bin/sh", f"for word in {looked_at}; do", '  case "$word" in']
        lines += [f'    {pattern}) if [ -e "{self.top / control}" ]; then exit 1; fi ;;' for pattern, control in rules.items()]
        lines += ["  esac", "done", f'exec "{shutil.which(tool)}" "$@"', ""]
        (self.s2bin / tool).write_text("\n".join(lines), encoding="utf-8")
        (self.s2bin / tool).chmod(0o755)

    def _clock_tools(self) -> None:
        log, clock = self.top / "calls.jsonl", self.top / "clock"
        leak = f'echo "leak: {SECRET}" >&2'
        (self.bin / "date").write_text(
            f'#!/bin/sh\n[ "$*" = "+%s" ] || exit 98\nread now < "{clock}"\n'
            f'echo \'{{"call": "date", "argv": ["date", "+%s"]}}\' >> "{log}"\necho "$now"\n{leak}\n',
            encoding="utf-8",
        )
        (self.bin / "sleep").write_text(
            f'#!/bin/sh\nread now < "{clock}"\necho $((now + $1)) > "{clock}.$$.tmp" && "{shutil.which("mv")}" '
            f'"{clock}.$$.tmp" "{clock}"\nprintf \'{{"call": "sleep", "argv": ["sleep", "%s"]}}\\n\' "$1" >> "{log}"\n'
            f'echo "leak: {SECRET}"\n{leak}\n',
            encoding="utf-8",
        )
        # logs the seconds it was given and the name of the wrapped command, prints nothing, then runs it
        (self.bin / "timeout").write_text(
            f'#!/bin/sh\nread now < "{clock}"\n'
            f'printf \'{{"call": "timeout", "argv": ["timeout", "%s", "%s"], "now": %s}}\\n\' "$1" "${{2##*/}}" "$now" >> "{log}"\n'
            'case $1 in "" | *[!0-9]* | 0*) exit 125 ;; esac\nshift\nexec "$@"\n',
            encoding="utf-8",
        )
        for tool in ("date", "sleep", "timeout"):
            (self.bin / tool).chmod(0o755)

    def _names(self) -> None:
        section, found = "", {}
        for line in (self.scripts / "stack.env.template").read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("["):
                section = line.strip("[]")
            else:
                found.setdefault(section, []).append(line)
        self.containers = found["containers"]
        self.production = found["production-container"][0]
        self.proxy_c, self.app_c, self.dash_c, self.gateway_c, self.browser_c = self.containers

    def seed_fs(self) -> None:
        """The two containers of Server 2 as folders: the owner app holds the page check, the browser an old copy."""
        shutil.rmtree(self.fs, ignore_errors=True)
        source = self.fs / self.app_c / "opt" / "raphael-page-check"
        (source / "lib").mkdir(parents=True)
        (source / "journey-run.sh").write_text("NEW run\n", encoding="utf-8")
        (source / "lib" / "helper.js").write_text("NEW helper\n", encoding="utf-8")
        (source / ".hidden").write_text("NEW hidden\n", encoding="utf-8")
        self.old_copy.mkdir(parents=True)
        (self.old_copy / "old.txt").write_text("OLD\n", encoding="utf-8")

    @property
    def source_folder(self) -> Path:
        return self.fs / self.app_c / "opt" / "raphael-page-check"

    @property
    def old_copy(self) -> Path:
        return self.fs / self.browser_c / "opt" / "raphael-journeys" / "tools" / "page-check"

    @property
    def staging(self) -> Path:
        return Path(f"{self.s2dir}.page-check.tar")

    def set_marker(self, text: str | None, folder: bool = False) -> None:
        """What the shipped folder on Server 2 holds in .refresh-commit: a text, nothing, or (unreadable) a folder."""
        self.s2dir.mkdir(parents=True, exist_ok=True)
        target = self.s2dir / ".refresh-commit"
        if folder:
            target.mkdir()
        elif text is not None:
            target.write_text(text, encoding="utf-8")

    def unreachable(self, how: str) -> None:
        """Server 2 states from which the marker path cannot be examined (as root, so with no permission bits): the
        folder is a plain file or a link that loops, or the marker is a link to nothing."""
        if how == "marker is a dangling link":
            self.s2dir.mkdir(parents=True, exist_ok=True)
            marker = self.s2dir / ".refresh-commit"
            marker.unlink(missing_ok=True)
            marker.symlink_to(self.top / "nowhere")
            return
        shutil.rmtree(self.s2dir, ignore_errors=True)
        if how == "folder is a plain file":
            self.s2dir.write_text("not a folder\n", encoding="utf-8")
        else:
            self.s2dir.symlink_to(self.s2dir)

    def snapshot(self) -> dict:
        """Everything on the fake Server 2 and its state, staging names included, for a before and after compare."""
        parent = self.s2dir.parent
        path = self.top / "state.json"
        state = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
            "images": self.answers["images"], "untagged": self.answers["untagged"], "tags": self.answers["tags"],
            "built_at": None, "converged": False,
        }
        state.pop("count", None)
        found = {}
        for item in sorted(parent.rglob("*")):
            relative = str(item.relative_to(parent))
            found[relative] = "link" if item.is_symlink() else ("folder" if item.is_dir() else item.read_bytes())
        for item in sorted(self.fs.rglob("*")):
            found["fs/" + str(item.relative_to(self.fs))] = "folder" if item.is_dir() else item.read_bytes()
        return {"tree": found, "state": state}

    def edit_template(self, old: str, new: str) -> None:
        path = self.scripts / "stack.env.template"
        path.write_text(path.read_text(encoding="utf-8").replace(old, new), encoding="utf-8")
        self._names()
        self.answer(browser=self.browser_c)
        self.seed_fs()

    def write_state(self, text: str | None, folder: Path | None = None) -> None:
        folder = folder or self.home
        if text is None:
            return
        (folder / "gateway_state.json").write_text(json.dumps({"pid": 7, "code_sha": text}), encoding="utf-8")

    def answer(self, **changes) -> None:
        for key, value in changes.items():
            if key in ("rc", "out", "advance", "late_rc", "cut", "garbled"):
                value = {k: v for k, v in {**self.answers[key], **value}.items() if v is not None}
            elif key == "inspect":
                value = {**self.answers[key], **value}
            self.answers[key] = value
        (self.top / "answers.json").write_text(json.dumps(self.answers), encoding="utf-8")

    def env(self, **changes: str | None) -> dict[str, str]:
        env = {
            "PATH": str(self.bin), "HOME": str(self.top), "HERMES_HOME": str(self.home),
            "JOURNEY_RUNNER_SSH": RUNNER, "JOURNEY_S2_SCRIPTS_DIR": str(self.s2dir),
            # values in the caller's environment that no call may carry along
            "CLAUDE_CODE_OAUTH_TOKEN": SECRET, "API_SERVER_KEY": SECRET, "ANTHROPIC_TOKEN": SECRET,
        }
        for name, value in changes.items():
            if value is None:
                env.pop(name, None)
            else:
                env[name] = value
        return env

    def run(self, *argv: str, **changes: str | None) -> subprocess.CompletedProcess[str]:
        assert SCRIPT.is_file(), "scripts/journeys/refresh_stack.sh does not exist"
        copy = self.scripts / SCRIPT.name
        shutil.copy2(SCRIPT, copy)
        env = self.env(**changes)
        self.destination = env.get("JOURNEY_RUNNER_SSH")
        self.runs.append((len(self.calls()), int((self.top / "clock").read_text(encoding="utf-8"))))
        done = subprocess.run(
            [str(copy), *argv], env=env, input="STDIN-SENTINEL\n", capture_output=True,
            encoding="utf-8", text=True, timeout=180, check=False,
        )
        self.audit()
        return done

    def audit(self) -> None:
        """Holds for every call of every test: no value, no forbidden word, JOURNEY_RUNNER_SSH alone, and
        every git, docker and ssh call sits behind a timeout that was given exactly the seconds left of its own run."""
        calls = self.calls()
        for index, call in enumerate(calls):
            assert SECRET not in json.dumps(call), f"a value reached a call: {call['call']}"
            assert not call.get("secret_in_stdin"), "a value was streamed"
            words = forbidden(" ".join(call["argv"][1:]))
            assert not words, f"{words} in {call['argv']}"
            if call["argv"][0] == "ssh":
                assert call["nargs"] == 2 and call["dest"] == self.destination, call["argv"]
            if call["call"] == "timeout":
                seconds = call["argv"][1]
                assert re.fullmatch(r"[1-9][0-9]*", seconds), call["argv"]
                began = [clock for first, clock in self.runs if first <= index][-1]
                assert int(seconds) == BOX - (call["now"] - began), (call["argv"], call["now"] - began)
        tools = ("git", "docker", "ssh")
        wrapped = Counter(c["argv"][2] for c in calls if c["call"] == "timeout" and c["argv"][2] in tools)
        called = Counter(c["argv"][0] for c in calls if c["argv"][0] in tools)
        assert wrapped == called, ("a git, docker or ssh call ran without its own timeout", wrapped, called)

    def calls(self) -> list[dict]:
        log = self.top / "calls.jsonl"
        return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()] if log.exists() else []

    def ssh_calls(self) -> list[dict]:
        return [call for call in self.calls() if call["argv"][0] == "ssh"]

    def kinds(self) -> list[str]:
        return [call["call"] for call in self.ssh_calls()]

    def last_kinds(self) -> list[str]:
        """The kinds of the ssh calls of the last run only (kinds() holds those of every run)."""
        return [call["call"] for call in self.calls()[self.runs[-1][0]:] if call["argv"][0] == "ssh"]

    def of(self, kind: str) -> list[dict]:
        return [call for call in self.calls() if call["call"] == kind]

    def remote(self, kind: str) -> list[str]:
        return [call["remote"] for call in self.of(kind)]

    def server1(self) -> list[str]:
        return [call["call"] for call in self.calls() if call["argv"][0] in ("git", "docker")]

    def sleeps(self) -> list[str]:
        return [call["argv"][1] for call in self.of("sleep")]

    def timeouts(self) -> list[tuple[str, str, int]]:
        """(wrapped command, seconds, seconds already used) of every timeout call, in order."""
        return [(c["argv"][2], c["argv"][1], c["now"] - CLOCK_START) for c in self.of("timeout")]

    def elapsed(self) -> int:
        return int((self.top / "clock").read_text(encoding="utf-8")) - CLOCK_START

    def server2(self) -> dict:
        """What is active on the fake Server 2: every file and folder beside the scripts folder (names of a staging
        kind left out), the image names (a loaded image id with no tag is not one), the retention tags, the build
        and converge marks, and the page check of the browser."""
        staging = (".new", ".part", ".page-check.tar")
        parent = self.s2dir.parent
        path = self.top / "state.json"
        state = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
            "images": self.answers["images"], "tags": self.answers["tags"], "built_at": None, "converged": False,
        }
        found = [p for p in sorted(parent.rglob("*")) if not any(part.endswith(staging) for part in p.relative_to(parent).parts)]
        return {
            "files": {str(p.relative_to(parent)): p.read_bytes() for p in found if p.is_file()},
            "folders": [str(p.relative_to(parent)) for p in found if p.is_dir()],
            "refs": sorted({i for i in state["images"] if not i.startswith("sha256:")}), "tags": state["tags"],
            "built": state["built_at"], "converged": state["converged"],
            "page": tree(self.old_copy) if self.old_copy.exists() else None,
        }

    def stdin_calls(self) -> list[str]:
        return [call["call"] for call in self.ssh_calls() if call["stdin_size"]]

    def cmd(self, tail: str) -> str:
        return f"cd '{self.s2dir}' && {tail}"

    def marker_cmd(self) -> str:
        """The marker read: the folder is searched and has the entry (cat it) or has not (3), or there is no folder and
        its parent can be searched (3); every other state, where the path cannot be examined, is 4."""
        folder, parent, marker = f"'{self.s2dir}'", f"'{self.s2dir.parent}'", f"'{self.s2dir}/.refresh-commit'"
        return (
            f"if [ -d {folder} ] && [ -x {folder} ]; then if [ -e {marker} ] || [ -L {marker} ]; then cat {marker}; "
            f"else exit 3; fi; elif [ -d {parent} ] && [ -x {parent} ] && [ ! -e {folder} ] && [ ! -L {folder} ]; "
            "then exit 3; else exit 4; fi"
        )

    def archive(self, commit: str = COMMIT) -> str:
        return f"{self.s2dir}.platform-{commit}.tar"

    def leftovers(self) -> list[str]:
        """Every staged name a failed run can leave on the fake Server 2, as far as it exists: beside the scripts folder
        (the kept .previous copy is no leftover) and beside the page check in the browser container."""
        found = [p.name for p in self.s2dir.parent.iterdir()
                 if p.name.startswith(self.s2dir.name + ".") and p.name != self.s2dir.name + ".previous"]
        beside = self.old_copy.parent
        found += [p.name for p in beside.iterdir() if p.name.startswith("page-check.")] if beside.exists() else []
        return sorted(found)

    def plant_leftovers(self) -> list[str]:
        """What failed runs leave behind, put there by hand: a staged scripts folder, the platform archive and its
        .part, the staging tar of the page check, and an old and a new sibling of the page check."""
        for folder in (Path(f"{self.s2dir}.new"), Path(f"{self.old_copy}.old"), Path(f"{self.old_copy}.new")):
            (folder / "inner").mkdir(parents=True)
            (folder / "inner" / "file.txt").write_text("LEFT\n", encoding="utf-8")
        for name in (self.archive(), f"{self.archive()}.part", str(self.staging)):
            Path(name).write_bytes(b"LEFT")
        return self.leftovers()

    def reconcile_cmds(self) -> dict[str, str]:
        """The four removals of the final reconciliation, one remote call each, by exact name: the staged scripts folder,
        the platform archive (and its .part), the page check .new and .old in the browser container, the staging tar."""
        new, archive, tar, page = f"{self.s2dir}.new", self.archive(), f"{self.s2dir}.page-check.tar", BROWSER_PAGE_CHECK
        loop = 'r=0; for d in "$@"; do if [ -e "$d" ]; then rm -r "$d" || r=1; fi; done; exit $r'
        return {
            "rm-scripts-new": f"if [ -e '{new}' ]; then rm -r '{new}'; fi",
            "rm-archive": (f"r=0; if [ -e '{archive}.part' ]; then rm '{archive}.part' || r=1; fi; "
                           f"if [ -e '{archive}' ]; then rm '{archive}' || r=1; fi; exit $r"),
            "rm-page-check": f"docker exec '{self.browser_c}' sh -c '{loop}' sh '{page}.new' '{page}.old'",
            "rm-staging-tar": f"if [ -e '{tar}' ]; then rm '{tar}'; fi",
        }

    def converge_cmd(self, proxy_tag: str = PROXY_TAG) -> str:
        return self.cmd(f"./stack_up.sh converge '{proxy_tag}' '{REVISION}' '{COMMIT}' '{REVISION}'")


def _need_tools() -> None:
    if not all(shutil.which(tool) for tool in ("bash", "dirname", "sh", "tar", "mkdir", "rm", "mv", "cat")):
        pytest.skip("needs bash, tar and coreutils")


@pytest.fixture
def box(tmp_path: Path) -> Sandbox:
    _need_tools()
    return Sandbox(tmp_path)


def cached(box: Sandbox) -> None:
    """Server 2 holds the scripts of this commit and both images; only the stack is stale."""
    box.set_marker(COMMIT + "\n")
    box.answer(images=[IMAGE_ID, PLATFORM_REF])


def reconciled(box: Sandbox) -> bool:
    """Each removal of the final reconciliation ran exactly once, with its exact text."""
    return {kind: box.remote(kind) for kind in RECONCILE} == {kind: [cmd] for kind, cmd in box.reconcile_cmds().items()}


def current(box: Sandbox) -> None:
    box.set_marker(COMMIT + "\n")
    box.answer(out={"versions": versions_text()})


def no_platform(box: Sandbox) -> None:
    box.set_marker(COMMIT + "\n")
    box.answer(images=[IMAGE_ID])


OLD = {n: str(n) * 40 for n in (1, 2, 3, 4)}
RETENTION_TAGS = {
    "raphael-journeys-proxy": ["a" * 12, "b" * 12, "c" * 12, PROXY_TAG, "d" * 12],
    "raphael-journeys-app-base": [OLD[1], OLD[2], OLD[3], REVISION, OLD[4]],
    "raphael-journeys-app": [f"{OLD[1]}-{CA}", f"{OLD[2]}-{CA}", f"{OLD[3]}-{CA}", APP_TAG,
                             f"{REVISION}-{'b' * 12}", f"{OLD[4]}-{CA}"],
    "raphael-journeys-platform": ["<none>", "bad tag!", "p1" * 20, "p2" * 20, "p3" * 20, "x" * 64, COMMIT, "p4" * 20],
    "raphael-journeys-browser": [f"{OLD[1]}-{CA}", APP_TAG],
}
REMOVED = {
    "raphael-journeys-proxy": ["c" * 12, "d" * 12],
    "raphael-journeys-app-base": [OLD[3], REVISION, OLD[4]],
    "raphael-journeys-app": [f"{OLD[3]}-{CA}", f"{REVISION}-{'b' * 12}", f"{OLD[4]}-{CA}"],
    "raphael-journeys-platform": ["p3" * 20, "p4" * 20],
    "raphael-journeys-browser": [],
}


def with_tags(box: Sandbox) -> None:
    current(box)
    box.answer(tags=RETENTION_TAGS)


def with_advances(box: Sandbox) -> None:
    box.answer(advance={"marker": 300, "ship-scripts": 120, "load": 60, "build-platform": 45, "converge": 200})


SCENARIOS = {
    "full": lambda box: None, "cached": cached, "current": current, "no-platform": no_platform,
    "tags": with_tags, "advances": with_advances,
}


@pytest.fixture(scope="module")
def shared(tmp_path_factory):
    """One run of the script per scenario for the whole module; tests only read what it left behind."""
    runs: dict[str, tuple[Sandbox, subprocess.CompletedProcess[str]]] = {}

    def get(name: str) -> tuple[Sandbox, subprocess.CompletedProcess[str]]:
        if name not in runs:
            _need_tools()
            sandbox = Sandbox(tmp_path_factory.mktemp(name))
            SCENARIOS[name](sandbox)
            runs[name] = (sandbox, sandbox.run())
        return runs[name]

    return get


# ---- refusals before any Server 2 step ------------------------------------------------------------


def test_script_is_executable_with_mode_755():
    assert SCRIPT.is_file(), "scripts/journeys/refresh_stack.sh does not exist"
    assert stat.S_IMODE(SCRIPT.stat().st_mode) == 0o755


def test_forbidden_word_check_sees_force_flags_and_removals():
    assert SCRIPT.is_file(), "scripts/journeys/refresh_stack.sh does not exist"
    assert forbidden("docker image rm -f 'a:b'") == ["-f"]
    assert forbidden("docker rm x") == ["docker rm"] and forbidden("docker container rm x") == ["container rm"]
    assert forbidden("docker system prune") == ["prune"] and forbidden("docker volume ls") == ["volume"]
    assert forbidden("sh -c 'rm -rf x --force'") == ["--force"]
    assert forbidden("tar -xf - -C 'd' --strip-components=2") == []
    assert forbidden("docker image rm 'raphael-journeys-app:tag'") == []


def test_an_argument_is_refused_without_any_call(box):
    expect_failure(box.run("now"), "usage")
    assert box.calls() == []


@pytest.mark.parametrize("changes", [
    {"JOURNEY_RUNNER_SSH": None}, {"JOURNEY_RUNNER_SSH": ""}, {"JOURNEY_RUNNER_SSH": "-oProxyCommand=x"},
    {"JOURNEY_RUNNER_SSH": "runner server"}, {"JOURNEY_RUNNER_SSH": "run;er"},
    {"JOURNEY_S2_SCRIPTS_DIR": "relative/dir"}, {"JOURNEY_S2_SCRIPTS_DIR": "/"}, {"JOURNEY_S2_SCRIPTS_DIR": "//"},
    {"JOURNEY_S2_SCRIPTS_DIR": "/dir with space"}, {"JOURNEY_S2_SCRIPTS_DIR": "/dir'quote"},
    {"JOURNEY_S2_SCRIPTS_DIR": "/dir;x"},
])
def test_invalid_settings_refuse_before_any_call_even_the_clock(box, changes):
    expect_failure(box.run(**changes), "config")
    assert box.calls() == []


@pytest.mark.parametrize("edit", [
    "missing", "not plain", "outside a section", "four containers", "no production", "two production",
    "odd production name",
])
def test_template_problems_refuse_before_any_call_even_the_clock(box, edit):
    path = box.scripts / "stack.env.template"
    text = path.read_text(encoding="utf-8")
    if edit == "missing":
        path.unlink()
    else:
        text = {
            "not plain": text.replace("raphael-workspace", "raphael-workspace=oops"),
            "outside a section": "stray-name\n" + text,
            "four containers": text.replace("raphael-journeys-browser\n\n[volumes]", "\n[volumes]"),
            "no production": text.replace("raphael-workspace\n", ""),
            "two production": text.replace("raphael-workspace\n", "raphael-workspace\nother-workspace\n"),
            "odd production name": text.replace("raphael-workspace", "-workspace"),
        }[edit]
        path.write_text(text, encoding="utf-8")
    expect_failure(box.run(), "config")
    assert box.calls() == []


def test_the_clock_is_read_right_after_the_checks_and_before_the_first_git_call(shared):
    box, _ = shared("full")
    calls = box.calls()
    git = [c["call"] for c in calls].index("git toplevel")
    assert git >= 2 and {c["call"] for c in calls[:git - 1]} == {"date"}
    assert calls[git - 1]["call"] == "timeout" and calls[git - 1]["argv"][2] == "git"


def test_a_trailing_slash_of_the_folder_is_stripped(box):
    box.answer(rc={"ship-scripts": 1})
    box.run(JOURNEY_S2_SCRIPTS_DIR=f"{box.s2dir}/")
    assert box.remote("marker") == [box.marker_cmd()]


@pytest.mark.parametrize("answers", [
    {"rc": {"git toplevel": 1}}, {"rc": {"git head": 1}}, {"head": "main"}, {"head": COMMIT.upper()},
    {"head": COMMIT[:39]}, {"head": COMMIT + "0"},
])
def test_unreadable_commit_refuses_before_any_ssh_call(box, answers):
    box.answer(**answers)
    expect_failure(box.run(), "commit")
    assert box.ssh_calls() == []


def test_git_is_asked_from_the_script_folder_and_archives_run_from_the_checkout_root(shared):
    box, done = shared("full")
    expect_success(done)
    assert [c["dir"] for c in box.of("git toplevel")] == [str(box.scripts)]
    assert [c["dir"] for c in box.of("git head")] in ([str(box.scripts)], [str(box.checkout)])
    archives = box.of("git archive scripts") + box.of("git archive commit")
    assert len(archives) == 2 and {c["dir"] for c in archives} == {str(box.checkout)}


GATEWAY_BAD = {
    "empty": "", "not json": "this is not json", "truncated": '{"pid": 7, "code_sha": ',
    "null": '{"pid": 7, "code_sha": null}', "number": '{"code_sha": 12345}', "short": '{"code_sha": "c0ffee"}',
    "uppercase": '{"code_sha": "%s"}' % COMMIT.upper(), "no key": '{"pid": 7}', "missing file": None,
}


@pytest.mark.parametrize("text", list(GATEWAY_BAD.values()), ids=list(GATEWAY_BAD))
def test_gateway_code_version_unreadable_or_absent_refuses_before_any_ssh_call(box, text):
    path = box.home / "gateway_state.json"
    if text is None:
        path.unlink()
    else:
        path.write_text(text, encoding="utf-8")
    expect_failure(box.run(), "gateway")
    assert box.ssh_calls() == []


def test_commit_and_gateway_code_version_that_differ_refuse_before_any_ssh_call(box):
    box.write_state(OTHER_COMMIT)
    expect_failure(box.run(), "differ")
    assert box.ssh_calls() == []


@pytest.mark.parametrize("text", [
    '{"code_sha":"%s"}', '{\n  "pid": 7,\n  "code_sha": "%s",\n  "kind": "hermes-gateway"\n}\n',
    '{"a": 1, "code_sha"  :   "%s"}',
])
def test_a_matching_gateway_code_version_starts_the_run(box, text):
    (box.home / "gateway_state.json").write_text(text % COMMIT, encoding="utf-8")
    box.answer(rc={"marker": 1, "ship-scripts": 1})
    box.run()
    assert box.kinds()[:1] == ["marker"]


def test_default_hermes_home_is_dot_hermes_under_home(box):
    (box.top / ".hermes").mkdir()
    box.write_state(COMMIT, box.top / ".hermes")
    (box.home / "gateway_state.json").unlink()
    box.answer(rc={"marker": 1, "ship-scripts": 1})
    box.run(HERMES_HOME=None)
    assert box.kinds()[:1] == ["marker"]


GOOD = f"{REVISION} {IMAGE_ID}\n"
BAD_OWNER = {
    "short label": (f"{'a' * 39} {IMAGE_ID}\n", 0, "revision"), "long label": (f"{'a' * 41} {IMAGE_ID}\n", 0, "revision"),
    "uppercase label": (f"{REVISION.upper()} {IMAGE_ID}\n", 0, "revision"),
    "missing label": (f" {IMAGE_ID}\n", 0, "revision"), "non-hex label": (f"{'g' * 40} {IMAGE_ID}\n", 0, "revision"),
    "short id": (f"{REVISION} sha256:abc\n", 0, "owner"), "uppercase id": (f"{REVISION} sha256:{'AB' * 32}\n", 0, "owner"),
    "id without prefix": (f"{REVISION} {'ab' * 32}\n", 0, "owner"), "empty": ("", 0, "owner"),
    "nonsense": ("nonsense\n", 0, "owner"), "two lines": (GOOD + GOOD, 0, "owner"),
    "extra field": (f"{REVISION} {IMAGE_ID} extra\n", 0, "owner"),
    "failing inspect": (GOOD, 1, "owner"), "failing inspect, no output": ("", 125, "owner"),
}


@pytest.mark.parametrize("out, rc, key", list(BAD_OWNER.values()), ids=list(BAD_OWNER))
def test_unusable_owner_app_refuses_before_any_ssh_call(box, out, rc, key):
    box.answer(inspect={"rc": rc, "out": out})
    expect_failure(box.run(), key)
    assert box.ssh_calls() == []


def test_server1_runs_only_docker_inspect_and_docker_save_of_the_inspected_id(shared):
    box, done = shared("full")
    expect_success(done)
    assert box.server1() == [
        "git toplevel", "git head", "docker inspect", "git archive scripts", "docker save", "git archive commit",
    ]
    (inspect,) = box.of("docker inspect")
    argv = inspect["argv"][1:]
    assert argv[:3] == ["inspect", "--type", "container"] and len(argv) == 6 and argv[-1] == box.production
    fmt = argv[argv.index("--format") + 1]
    assert "org.opencontainers.image.revision" in fmt and ".Image" in fmt and "Env" not in fmt
    assert [c["argv"] for c in box.of("docker save")] == [["docker", "save", IMAGE_ID]]


def test_production_container_name_comes_from_the_template(box):
    box.edit_template("raphael-workspace", "renamed-production")
    box.answer(rc={"marker": 1, "ship-scripts": 1})
    box.run()
    assert box.of("docker inspect")[0]["argv"][-1] == "renamed-production"


# ---- first look: three read-only calls, stopping at the first "not current" -------------------------


def test_a_current_stack_crosses_nothing_and_goes_straight_to_the_retention(shared):
    box, done = shared("current")
    expect_success(done)
    assert box.kinds() == CURRENT_RUN
    assert box.remote("marker") == [box.marker_cmd()]
    assert box.remote("versions") == [box.cmd("./stack_up.sh versions")] * 2
    assert box.remote("page-check-present") == [f"docker exec '{box.browser_c}' test -d '{BROWSER_PAGE_CHECK}'"]
    assert box.server1() == ["git toplevel", "git head", "docker inspect"]
    assert all(call["stdin_size"] == 0 for call in box.ssh_calls())
    assert box.sleeps() == [] and reconciled(box)
    assert sorted(path.name for path in box.s2dir.parent.iterdir()) == ["raphael-journeys"]
    assert tree(box.s2dir) == {".refresh-commit": COMMIT + "\n"}


def test_matching_versions_without_the_page_check_are_not_current(box):
    current(box)
    box.answer(rc={"page-check-present": 1, "has-image": 2})
    expect_failure(box.run(), "image-ask")
    assert box.kinds() == ["marker", "versions", "page-check-present", "has-image"]
    assert box.of("git archive scripts") == []


def _swapped() -> str:
    lines = versions_text().splitlines(keepends=True)
    return "".join([lines[1], lines[0], *lines[2:]])


NOT_CURRENT = {
    "versions fails": (versions_text(), 1), "ssh fails": (versions_text(), 255), "garbled": ("garbage\n", 0),
    "empty": ("", 0), "four lines": (versions_text().rsplit("browser", 1)[0], 0),
    "extra line": (versions_text() + "extra x\n", 0), "wrong order": (_swapped(), 0),
    "blank line": (versions_text().replace("app ", "\napp "), 0), "unknown role": (versions_text().replace("gateway", "gatewa"), 0),
    "proxy none": (versions_text(proxy="none"), 0), "proxy not plain": (versions_text(proxy="a b"), 0),
    "dashboard differs": (versions_text(dashboard=OTHER_COMMIT), 0),
    "gateway differs": (versions_text(gateway=OTHER_COMMIT), 0),
    "app other revision": (versions_text(app=f"{OTHER_COMMIT}-{CA}"), 0),
    "app without hash": (versions_text(app=REVISION), 0),
    "app uppercase hex": (versions_text(app=f"{REVISION}-{CA.upper()}"), 0),
    "app short hex": (versions_text(app=f"{REVISION}-{CA[:11]}"), 0),
    "browser other revision": (versions_text(browser=f"{OTHER_COMMIT}-{CA}"), 0),
    "browser none": (versions_text(browser="none"), 0),
}


@pytest.mark.parametrize("text, rc", list(NOT_CURRENT.values()), ids=list(NOT_CURRENT))
def test_versions_that_differ_or_cannot_be_read_are_not_current_and_the_look_stops_there(box, text, rc):
    box.set_marker(COMMIT + "\n")
    box.answer(out={"versions": text}, rc={"versions": rc, "has-image": 2})
    expect_failure(box.run(), "image-ask")
    assert box.kinds() == ["marker", "versions", "has-image"]


MARKERS = {"missing": None, "other commit": OTHER_COMMIT + "\n", "other commit without a newline": OTHER_COMMIT}


@pytest.mark.parametrize("text", list(MARKERS.values()), ids=list(MARKERS))
def test_a_marker_that_was_read_and_differs_or_is_absent_is_not_current_and_ships_the_scripts_once(box, text):
    box.set_marker(text)
    box.answer(rc={"ship-scripts": 1})
    expect_failure(box.run(), "ship-scripts")
    assert box.kinds() == [*GATE, "ship-scripts"]
    assert box.stdin_calls() == ["ship-scripts"] and box.server1()[-1] == "git archive scripts"
    assert len(box.of("git archive scripts")) == 1


GARBLED_MARKERS = {
    "empty": "", "only a newline": "\n", "39 characters": OTHER_COMMIT[:39] + "\n", "41 characters": OTHER_COMMIT + "0\n",
    "upper case": OTHER_COMMIT.upper() + "\n", "not hex": "z" * 40 + "\n", "two commit lines": COMMIT + "\n" + COMMIT + "\n",
    "two different commit lines": OTHER_COMMIT + "\n" + COMMIT + "\n", "a blank line after the commit": OTHER_COMMIT + "\n\n",
    "text after the commit": OTHER_COMMIT + " extra\n", "a carriage return": OTHER_COMMIT + "\r\n",
    "equal commit and text": COMMIT + " extra\n",
}


@pytest.mark.parametrize("text", list(GARBLED_MARKERS.values()), ids=list(GARBLED_MARKERS))
def test_a_marker_that_is_not_one_line_of_forty_lowercase_hex_digits_ends_the_run_before_any_transfer(box, text):
    cached(box)
    box.set_marker(text)
    before = box.snapshot()
    expect_failure(box.run(), "marker-ask")
    assert box.kinds() == ["marker"] and box.stdin_calls() == [] and not SHIPPING & set(box.kinds())
    assert box.server1() == ["git toplevel", "git head", "docker inspect"] and box.snapshot() == before


@pytest.mark.parametrize("how", ["folder is a plain file", "folder is a link that loops", "marker is a dangling link"])
def test_a_marker_path_that_cannot_be_examined_is_a_failed_read_and_not_an_absent_marker(box, how):
    cached(box)
    box.unreachable(how)
    before = box.snapshot()
    expect_failure(box.run(), "marker-ask")
    assert box.kinds() == ["marker"] and box.stdin_calls() == [] and box.snapshot() == before
    assert box.server1() == ["git toplevel", "git head", "docker inspect"]


def test_a_parent_folder_that_cannot_be_searched_is_no_evidence_that_the_marker_is_absent(box):
    missing = box.top / "no-such-parent" / "raphael-journeys"
    box.answer(s2_dir=str(missing))
    expect_failure(box.run(JOURNEY_S2_SCRIPTS_DIR=str(missing)), "marker-ask")
    assert box.kinds() == ["marker"] and box.stdin_calls() == [] and not missing.parent.exists()


UNREADABLE_MARKERS = {**{f"status {rc}": rc for rc in (1, 2, 4, 125, 126, 127, 255)}, "a folder": None}


@pytest.mark.parametrize("rc", list(UNREADABLE_MARKERS.values()), ids=list(UNREADABLE_MARKERS))
def test_a_marker_that_could_not_be_read_ends_the_run_and_nothing_crosses_or_changes(box, rc):
    cached(box)
    if rc is None:
        (box.s2dir / ".refresh-commit").unlink()
        box.set_marker(None, folder=True)
    else:
        box.answer(rc={"marker": rc})
    before = box.server2()
    expect_failure(box.run(), "marker-ask")
    assert box.kinds() == ["marker"] and box.stdin_calls() == []
    assert box.server1() == ["git toplevel", "git head", "docker inspect"] and box.server2() == before


def test_the_marker_one_run_leaves_makes_the_next_run_find_the_stack_current(box):
    expect_success(box.run())
    assert box.stdin_calls() == ["ship-scripts", "load", "ship-platform"]
    before = len(box.ssh_calls())
    box.answer(out={"versions": versions_text()})
    state = json.loads((box.top / "state.json").read_text(encoding="utf-8"))
    state["count"] = {}
    (box.top / "state.json").write_text(json.dumps(state), encoding="utf-8")
    expect_success(box.run())
    assert [c["call"] for c in box.ssh_calls()[before:]] == CURRENT_RUN
    assert box.stdin_calls() == ["ship-scripts", "load", "ship-platform"]
    assert len(box.of("git archive scripts")) == 1 and len(box.of("docker save")) == 1


# ---- a. the scripts folder is shipped and swapped in -----------------------------------------------


def expected_scripts(box: Sandbox) -> dict[str, str]:
    return {p.removeprefix("scripts/journeys/"): t for p, t in box.answers["scripts_files"].items()}


def test_scripts_are_shipped_first_among_the_shipping_steps(shared):
    box, _ = shared("full")
    kinds = box.kinds()
    assert kinds[:5] == [*GATE, "ship-scripts", "publish-scripts"]
    assert 4 < min(kinds.index(k) for k in ("load", "ship-platform", "build-platform"))
    (ship,) = box.of("ship-scripts")
    (archive,) = box.of("git archive scripts")
    assert ship["stdin_sha"] == archive["sha"] and ship["stdin_size"] > 0
    assert archive["argv"][1:] == ["-C", str(box.checkout), "archive", "--format=tar", COMMIT, "scripts/journeys"]


def test_a_first_run_without_a_folder_creates_it_with_the_files_and_the_marker_directly_inside(shared):
    box, _ = shared("full")
    parent = box.s2dir.parent
    assert tree(box.s2dir) == {**expected_scripts(box), ".refresh-commit": COMMIT + "\n"}
    assert os.access(box.s2dir / "stack_up.sh", os.X_OK)
    assert not (parent / "raphael-journeys.new").exists() and not (parent / "raphael-journeys.previous").exists()


def test_the_stage_only_extracts_into_the_new_folder_and_the_publish_call_writes_the_marker_before_the_renames(shared):
    box, _ = shared("full")
    new = f"{box.s2dir}.new"
    assert box.kinds()[3:5] == ["ship-scripts", "publish-scripts"]
    (stage,) = box.remote("ship-scripts")
    (publish,) = box.remote("publish-scripts")
    write = f"echo '{COMMIT}' > '{new}/.refresh-commit'"
    assert stage == (f"if [ -e '{new}' ]; then rm -r '{new}'; fi && mkdir '{new}' && tar -xf - -C '{new}' --strip-components=2")
    assert publish.count(write) == 1 and publish.startswith(write) and publish.index(write) < publish.index("mv ")
    assert "tar -xf" not in publish and publish.endswith(f"else mv '{new}' '{box.s2dir}'; fi")
    assert [c["stdin_size"] > 0 for c in (box.of("ship-scripts")[0], box.of("publish-scripts")[0])] == [True, False]


def test_an_existing_folder_is_kept_as_the_previous_copy_and_the_new_one_holds_the_marker(box):
    parent = box.s2dir.parent
    box.set_marker(OTHER_COMMIT + "\n")
    (box.s2dir / "old.txt").write_text("OLD\n", encoding="utf-8")
    box.answer(rc={"load": 1})
    expect_failure(box.run(), "app-image")
    assert tree(parent / "raphael-journeys.previous") == {"old.txt": "OLD\n", ".refresh-commit": OTHER_COMMIT + "\n"}
    assert tree(box.s2dir) == {**expected_scripts(box), ".refresh-commit": COMMIT + "\n"}
    assert not (parent / "raphael-journeys.new").exists()


def test_a_second_refresh_replaces_the_previous_copy_and_a_stale_new_folder(box):
    parent = box.s2dir.parent
    box.set_marker(OTHER_COMMIT + "\n")
    (box.s2dir / "gen1.txt").write_text("ONE\n", encoding="utf-8")
    for suffix, name, text in (("previous", "ancient.txt", "ANCIENT\n"), ("new", "junk.txt", "JUNK\n")):
        (parent / f"raphael-journeys.{suffix}").mkdir()
        (parent / f"raphael-journeys.{suffix}" / name).write_text(text, encoding="utf-8")
    box.answer(rc={"load": 1})
    expect_failure(box.run(), "app-image")
    assert tree(parent / "raphael-journeys.previous") == {"gen1.txt": "ONE\n", ".refresh-commit": OTHER_COMMIT + "\n"}
    assert tree(box.s2dir) == {**expected_scripts(box), ".refresh-commit": COMMIT + "\n"}
    assert not (parent / "raphael-journeys.new").exists()
    (box.s2dir / ".refresh-commit").write_text(OTHER_COMMIT + "\n", encoding="utf-8")
    shipped = tree(box.s2dir)
    box.answer(scripts_files={"scripts/journeys/stack_up.sh": "# third generation\n"})
    expect_failure(box.run(), "app-image")
    assert tree(parent / "raphael-journeys.previous") == shipped
    assert tree(box.s2dir) == {"stack_up.sh": "# third generation\n", ".refresh-commit": COMMIT + "\n"}


@pytest.mark.parametrize("rc, kinds", [
    ({"ship-scripts": 1}, [*GATE, "ship-scripts"]), ({"git archive scripts": 1}, [*GATE, "ship-scripts"]),
    ({"publish-scripts": 1}, [*GATE, "ship-scripts", "publish-scripts"]),
], ids=["receiver", "producer", "publish"])
def test_scripts_that_cannot_be_shipped_end_the_run(box, rc, kinds):
    box.answer(rc=rc)
    expect_failure(box.run(), "ship-scripts")
    assert box.kinds() == kinds and not box.s2dir.exists()


def test_a_real_failure_of_the_scripts_stage_ends_the_run_before_any_publish_and_changes_nothing(box):
    stuck = Path(f"{box.s2dir}.new")
    stuck.symlink_to(box.top / "nowhere")
    before = box.snapshot()
    expect_failure(box.run(), "ship-scripts")
    assert box.kinds() == [*GATE, "ship-scripts"] and not box.s2dir.exists() and box.snapshot() == before


def test_every_remote_command_parses_as_sh(shared):
    box, done = shared("full")
    expect_success(done)
    assert box.kinds() == FULL_RUN
    assert {"publish-scripts", "publish-platform", "copy-page-check", "publish-page-check"} <= set(box.kinds())
    for call in box.ssh_calls():
        words = shlex.split(call["remote"])
        inner = [words[i + 1] for i, word in enumerate(words[:-1]) if word == "-c"]
        for text in [call["remote"], *inner]:
            parsed = subprocess.run(["sh", "-n", "-c", text], capture_output=True, encoding="utf-8", text=True)
            assert parsed.returncode == 0, (text, parsed.stderr)


# ---- b. the owner app image: loaded only when Server 2 lacks the id --------------------------------


def test_a_known_app_image_is_not_loaded_but_still_tagged(shared):
    box, done = shared("no-platform")
    expect_success(done)
    assert "load" not in box.kinds() and "docker save" not in box.server1()
    assert box.remote("has-image")[0] == INSPECT_APP
    assert box.remote("tag") == [f"docker tag '{IMAGE_ID}' 'raphael-journeys-app-base:{REVISION}'"]
    assert box.stdin_calls() == ["ship-platform"]


def test_a_missing_app_image_is_saved_once_and_loaded_once_and_then_tagged(shared):
    box, _ = shared("full")
    (save,) = box.of("docker save")
    (load,) = box.of("load")
    assert save["argv"] == ["docker", "save", IMAGE_ID]
    assert load["remote"] == "docker load" and load["stdin_sha"] == save["sha"]
    assert load["stdin_size"] == len(box.answers["save_data"])
    kinds = box.kinds()
    assert kinds.index("has-image") < kinds.index("load") < kinds.index("tag")
    assert box.remote("tag") == [f"docker tag '{IMAGE_ID}' 'raphael-journeys-app-base:{REVISION}'"]
    assert box.stdin_calls() == ["ship-scripts", "load", "ship-platform"]


@pytest.mark.parametrize("rc", [{"load": 1}, {"tag": 1}, {"docker save": 1}], ids=["load", "tag", "save"])
def test_a_failed_load_tag_or_save_ends_with_its_reason(box, rc):
    box.answer(rc=rc)
    expect_failure(box.run(), "app-image")
    assert not {"build-platform", "converge"} & set(box.kinds())
    assert APP_BASE_TAG not in box.server2()["refs"], "the tag is made only after the whole image was loaded"


def test_an_id_that_docker_does_not_know_while_it_holds_other_images_is_missing_and_it_is_shipped(box):
    box.answer(images=["sha256:" + "ef56" * 16])
    expect_success(box.run())
    assert box.kinds() == FULL_RUN and box.remote("has-image")[0] == INSPECT_APP and len(box.of("load")) == 1


def test_an_app_image_id_held_without_a_tag_is_not_saved_or_loaded_again_and_only_gets_its_tag(box):
    cached(box)
    box.answer(untagged=[IMAGE_ID])
    expect_success(box.run())
    assert box.of("docker save") == [] and box.of("load") == [], "an id Server 2 holds without a tag was shipped again"
    assert box.kinds() == CACHED_RUN and box.remote("has-image") == [INSPECT_APP, INSPECT_PLATFORM]
    assert box.server1() == ["git toplevel", "git head", "docker inspect"] and box.stdin_calls() == []
    assert box.remote("tag") == [f"docker tag '{IMAGE_ID}' 'raphael-journeys-app-base:{REVISION}'"]
    assert APP_BASE_TAG in box.server2()["refs"]


# ---- b and c. a read that failed or was garbled is not "missing": nothing crosses ---------------------

NOT_FOUND_OTHER = "Error response from daemon: No such image: raphael-journeys-platform:" + "e" * 40 + "\n"
GARBLED_READS = {
    "not an id": "hello\n", "upper case hex": "sha256:" + "AB" * 32 + "\n", "63 hex digits": "sha256:" + "a" * 63 + "\n",
    "65 hex digits": "sha256:" + "a" * 65 + "\n", "no prefix": "ab" * 32 + "\n", "an id and a junk line": IMAGE_ID + "\njunk\n",
    "empty": "", "only a blank line": "\n", "an empty list": "[]\n",
}
UNREADABLE_READS = {
    **{f"status {rc}": {"rc": rc} for rc in (1, 2, 125, 255)},
    **{f"garbled, {name}": {"garbled": text} for name, text in GARBLED_READS.items()},
    "not-found line of another image": {"rc": 1, "garbled": NOT_FOUND_OTHER},
}


def break_read(box: Sandbox, key: str, case: dict) -> None:
    box.answer(**{what: {key: value} for what, value in case.items()})


@pytest.mark.parametrize("case", list(UNREADABLE_READS.values()), ids=list(UNREADABLE_READS))
def test_an_app_image_read_that_could_not_be_read_ends_the_run_and_nothing_crosses_or_changes(box, case):
    cached(box)
    break_read(box, "has-image#1", case)
    before = box.server2()
    expect_failure(box.run(), "image-ask")
    assert box.kinds() == CACHED_RUN[:3] and box.stdin_calls() == []
    assert box.server1() == ["git toplevel", "git head", "docker inspect"] and box.server2() == before


@pytest.mark.parametrize("case", list(UNREADABLE_READS.values()), ids=list(UNREADABLE_READS))
def test_a_platform_image_read_that_could_not_be_read_ends_the_run_and_nothing_crosses(box, case):
    cached(box)
    break_read(box, "has-image#2", case)
    before = box.snapshot()
    expect_failure(box.run(), "image-ask")
    assert box.kinds() == CACHED_RUN[:4] and box.stdin_calls() == []
    assert box.server1() == ["git toplevel", "git head", "docker inspect"] and box.snapshot() == before


@pytest.mark.parametrize("case", list(UNREADABLE_READS.values()), ids=list(UNREADABLE_READS))
def test_a_platform_image_read_that_could_not_be_read_in_the_wait_ends_the_run_and_the_build_keeps_running(box, case):
    no_platform(box)
    break_read(box, "has-image#3", case)
    expect_failure(box.run(), "image-ask")
    kinds = box.kinds()
    assert kinds[-2:] == ["build-platform", "has-image"] and not AFTER_BOX & set(kinds)
    assert box.server2()["built"] is not None


def test_an_app_image_read_that_answers_with_another_id_than_the_one_asked_is_no_find(box):
    cached(box)
    box.answer(garbled={"has-image#1": "sha256:" + "ef56" * 16 + "\n"})
    before = box.snapshot()
    expect_failure(box.run(), "image-ask")
    assert box.kinds() == CACHED_RUN[:3] and box.stdin_calls() == [] and box.snapshot() == before


def test_a_platform_image_read_may_answer_with_any_well_formed_id(box):
    cached(box)
    box.answer(platform_id="sha256:" + "ef56" * 16)
    expect_success(box.run())
    assert box.kinds() == CACHED_RUN and box.stdin_calls() == []


INITIAL_READS = {
    "app id read fails": (1, {"rc": 255}), "app id read is empty": (1, {"garbled": ""}),
    "app id answer is a not-found line of another image": (1, {"rc": 1, "garbled": NOT_FOUND_OTHER}),
    "platform read fails": (2, {"rc": 255}), "platform read is empty": (2, {"garbled": ""}),
    "platform answer is a not-found line of another image": (2, {"rc": 1, "garbled": NOT_FOUND_OTHER}),
}
STALE_STATES = {
    "stale scripts": lambda box: (cached(box), box.set_marker(OTHER_COMMIT + "\n")),
    "missing app image": lambda box: (cached(box), box.answer(images=[PLATFORM_REF])),
    "stale scripts and missing app image": lambda box: (box.set_marker(OTHER_COMMIT + "\n"), box.answer(images=[PLATFORM_REF])),
}


@pytest.mark.parametrize("state", list(STALE_STATES.values()), ids=list(STALE_STATES))
@pytest.mark.parametrize("which, case", list(INITIAL_READS.values()), ids=list(INITIAL_READS))
def test_an_initial_image_read_that_failed_or_was_garbled_ends_the_run_before_the_scripts_or_an_image_cross(
    box, which, case, state,
):
    state(box)
    break_read(box, f"has-image#{which}", case)
    before = box.snapshot()
    expect_failure(box.run(), "image-ask")
    kinds = box.kinds()
    assert kinds[-1] == "has-image" and kinds.count("has-image") == which and not SHIPPING & set(kinds)
    assert box.stdin_calls() == [] and box.server1() == ["git toplevel", "git head", "docker inspect"]
    assert box.snapshot() == before


@pytest.mark.parametrize("stdout, error", [
    ("\n", "Error response from daemon: No such image: %s\n"), ("[]\n", "Error response from daemon: No such image: %s\n"),
    ("", "Error response from daemon: No such image: %s\n"), ("\n", "Error: No such image: %s\n"),
    ("[]\n", "Error: No such image: %s\n"),
], ids=["empty line", "[]", "nothing on stdout", "empty line, old wording", "[], old wording"])
def test_docker_own_not_found_answer_is_accepted_in_both_shapes_it_has(box, stdout, error):
    box.answer(missing_stdout=stdout, missing_error=error)
    expect_success(box.run())
    assert box.kinds() == FULL_RUN and box.remote("has-image")[:2] == [INSPECT_APP, INSPECT_PLATFORM]
    assert box.remote("has-image")[0].endswith(" 2>&1")


@pytest.mark.parametrize("error", [
    "Error response from daemon: No such image: raphael-journeys-platform:" + "e" * 40 + "\n",
    "Error response from daemon: No such image: %s:more\n", "Error response from daemon: No such image: x%s\n",
    "Error response from daemon: No such image: %s \n", "error response from daemon: no such image: %s\n",
    "Error response from daemon: No such image: %s\nError response from daemon: No such image: %s\n",
    "Error response from daemon: Get \"https://registry/v2/\": no such host\n", "Error response from daemon: No such image\n",
], ids=["another image", "longer name", "name with a prefix", "trailing blank", "lower case", "two lines", "other error", "no name"])
def test_a_not_found_line_that_is_not_dockers_own_for_the_asked_name_is_no_absence(box, error):
    box.answer(missing_error=error)
    before = box.snapshot()
    expect_failure(box.run(), "image-ask")
    assert box.kinds() == [*GATE[:2]] and box.stdin_calls() == [] and box.snapshot() == before


# ---- c. the platform image ------------------------------------------------------------------------


def test_both_images_known_and_the_marker_equal_cross_nothing_and_the_stack_still_converges(shared):
    box, done = shared("cached")
    expect_success(done)
    assert box.kinds() == CACHED_RUN
    assert box.server1() == ["git toplevel", "git head", "docker inspect"]
    assert box.stdin_calls() == [] and not SHIPPING & set(box.kinds()) and reconciled(box)
    assert box.remote("has-image") == [INSPECT_APP, INSPECT_PLATFORM]
    assert box.remote("tag") == [f"docker tag '{IMAGE_ID}' 'raphael-journeys-app-base:{REVISION}'"]
    assert box.sleeps() == [] and not box.staging.parent.joinpath("raphael-journeys.new").exists()


def test_a_missing_platform_image_is_staged_as_one_file_under_a_temporary_name_published_by_a_rename_and_built(shared):
    box, done = shared("no-platform")
    expect_success(done)
    part, final = f"{box.archive()}.part", box.archive()
    assert box.remote("ship-platform") == [f"cat > '{part}'"]
    assert box.remote("publish-platform") == [f"mv '{part}' '{final}'"]
    assert box.remote("build-platform") == [box.cmd(f"./stack_up.sh build platform '{COMMIT}' '{final}'")]
    (ship,) = box.of("ship-platform")
    (git,) = box.of("git archive commit")
    assert ship["stdin_sha"] == git["sha"] and git["argv"][1:] == ["-C", str(box.checkout), "archive", "--format=tar", COMMIT]
    assert [c["stdin_size"] for c in box.of("publish-platform")] == [0]
    kinds = box.kinds()
    assert kinds.index("ship-platform") + 2 == kinds.index("publish-platform") + 1 == kinds.index("build-platform")
    assert kinds.count("build-platform") == 1
    assert box.server1() == ["git toplevel", "git head", "docker inspect", "git archive commit"]
    state = json.loads((box.top / "state.json").read_text(encoding="utf-8"))
    assert state["archive_sha"] == git["sha"], "the build must see the whole archive under its final name"


def test_the_wait_checks_at_once_then_every_ten_seconds_and_the_archive_is_left_for_the_final_reconciliation(box):
    no_platform(box)
    box.answer(ready_after=60, rc={"build-proxy": 1})
    expect_failure(box.run(), "proxy-tag")
    kinds = box.kinds()
    assert kinds[kinds.index("build-platform") + 1] == "has-image"
    # the two questions before the shipping, then the first look at once and one more look after each of the 6 sleeps
    assert box.remote("has-image")[2:] == [INSPECT_PLATFORM] * 8 and box.remote("has-image")[:2] == [INSPECT_APP, INSPECT_PLATFORM]
    assert box.sleeps() == ["10"] * 6 and box.elapsed() == 60
    assert kinds[-2:] == ["has-image", "build-proxy"] and "rm-archive" not in kinds
    assert Path(box.archive()).is_file(), "a failed run leaves the archive; the next run's reconciliation removes it"
    box.answer(rc={"build-proxy": None})
    expect_success(box.run())
    assert not Path(box.archive()).exists() and len(box.of("build-platform")) == 1


@pytest.mark.parametrize("rc, last", [
    ({"ship-platform": 1}, "ship-platform"), ({"publish-platform": 1}, "publish-platform"),
    ({"build-platform": 1}, "build-platform"), ({"git archive commit": 1}, "ship-platform"),
], ids=["ship", "publish", "build", "git"])
def test_a_platform_archive_or_build_that_fails_ends_the_run(box, rc, last):
    no_platform(box)
    box.answer(rc=rc)
    expect_failure(box.run(), "platform-build")
    assert box.kinds()[-1] == last and not {"build-proxy", "rm-archive", "converge"} & set(box.kinds())


# ---- d. proxy build, e. converge -------------------------------------------------------------------


def test_the_full_run_goes_in_the_order_of_the_plan_and_converge_gets_exactly_four_arguments(shared):
    box, done = shared("full")
    expect_success(done)
    assert box.kinds() == FULL_RUN
    assert box.remote("build-proxy") == [box.cmd("./stack_up.sh build proxy")]
    assert box.remote("converge") == [box.converge_cmd()]
    assert all(call["stdin_size"] == 0 for call in box.ssh_calls() if call["call"] in ("converge", "build-proxy"))


def test_a_proxy_tag_of_63_plain_characters_is_passed_on_to_converge(box):
    cached(box)
    tag = "a" * 30 + "." + "B-c_" * 8
    assert len(tag) == 63
    box.answer(out={"build-proxy": f"stack_up: built proxy {tag}\n"}, rc={"copy-page-check": 1})
    expect_failure(box.run(), "page-check")
    assert box.remote("converge") == [box.converge_cmd(tag)]


PROXY_BAD = {
    "no line": ("stack_up: built app 1\n", 0), "empty": ("", 0), "tag not plain": ("stack_up: built proxy a b\n", 0),
    "tag too long": (f"stack_up: built proxy {'a' * 64}\n", 0), "empty tag": ("stack_up: built proxy \n", 0),
    "slash in tag": ("stack_up: built proxy a/b\n", 0), "text before": (f"x stack_up: built proxy {PROXY_TAG}\n", 0),
    "nonzero exit": (f"stack_up: built proxy {PROXY_TAG}\n", 1),
}


@pytest.mark.parametrize("out, rc", list(PROXY_BAD.values()), ids=list(PROXY_BAD))
def test_a_proxy_build_that_reports_no_plain_tag_ends_with_its_reason(box, out, rc):
    cached(box)
    box.answer(out={"build-proxy": out}, rc={"build-proxy": rc})
    expect_failure(box.run(), "proxy-tag")
    assert "converge" not in box.kinds()


@pytest.mark.parametrize("out", [
    "stack_up: making\nstack_up: the stack answers\n",
    "stack_up: making\n\nstack_up: the first reset starts the owner app and the gateway\n\n\n",
    "stack_up: the stack answers",
], ids=["answers", "first reset, blank lines", "no final newline"])
def test_both_success_lines_of_converge_are_accepted(box, out):
    cached(box)
    box.answer(out={"converge": out}, rc={"copy-page-check": 1})
    expect_failure(box.run(), "page-check")
    assert box.kinds()[-2:] == ["converge", "copy-page-check"]


CONVERGE_BAD = {
    "other last line": ("stack_up: the stack answers\nstack_up: waiting\n", 0), "empty output": ("", 0),
    "nonzero exit": ("stack_up: the stack answers\n", 1), "trailing space": ("stack_up: the stack answers \n", 0),
    "prefix only": ("stack_up: the stack\n", 0),
    "failure line": ("stack_up: container x is not one this script made; it was left alone\n", 1),
}


@pytest.mark.parametrize("out, rc", list(CONVERGE_BAD.values()), ids=list(CONVERGE_BAD))
def test_a_converge_that_does_not_end_well_ends_with_its_reason(box, out, rc):
    cached(box)
    box.answer(out={"converge": out}, rc={"converge": rc})
    expect_failure(box.run(), "converge")
    assert box.kinds()[-1] == "converge"


# ---- f. the page check copy ------------------------------------------------------------------------


def test_the_page_check_is_staged_by_one_chain_without_a_pipe_and_swapped_by_a_second_call_before_the_reconciliation(shared):
    box, done = shared("cached")
    expect_success(done)
    kinds = box.kinds()
    at = kinds.index("converge")
    assert kinds[at + 1:at + 8] == ["copy-page-check", "publish-page-check", *RECONCILE, "versions"]
    (stage,) = box.remote("copy-page-check")
    (publish,) = box.remote("publish-page-check")
    tar, final = f"{box.s2dir}.page-check.tar", BROWSER_PAGE_CHECK
    exec_sh = f"docker exec '{box.browser_c}' sh -c "
    produce = f"docker cp '{box.app_c}:{OPT}/.' - > '{tar}'"
    consume = f"docker cp - '{box.browser_c}:{final}.new' < '{tar}'"
    assert stage.count("docker cp ") == 2 and stage.count(" && ") == 3 and stage.count(exec_sh) == 2
    assert not re.search(r"(?<!\|)\|(?!\|)", stage), "the stage must not be a pipeline"
    assert stage.startswith(exec_sh) and stage.index(produce) < stage.index(consume) < stage.rindex(exec_sh)
    assert stage.endswith(f" sh '{final}.new'") and not re.search(re.escape(final) + r"(?!\.(new|old)')", stage)
    assert "mv " not in stage and "rm " not in stage.split(produce)[1], "the stage must not swap or clean anything"
    assert publish.startswith(exec_sh) and publish.endswith(f" sh '{final}.new' '{final}' '{final}.old'")
    assert " && " not in publish and "docker cp" not in publish and tar not in publish, "the swap cleans nothing"
    assert publish.index('rm -r "$o"') < publish.index('mv "$f" "$o"'), "an old sibling is removed first, to make room"
    assert publish.count("rm ") == 1, "after the swap the old copy stays for the final reconciliation"
    assert [c["stdin_size"] for c in box.of("copy-page-check") + box.of("publish-page-check")] == [0, 0]


def test_the_copy_replaces_the_old_page_check_with_the_whole_new_one_and_leaves_no_sibling_or_staging_file(shared):
    box, _ = shared("cached")
    source = tree(box.source_folder)
    assert tree(box.old_copy) == source and ".hidden" in source and "lib/helper.js" in source and "old.txt" not in source
    assert [path.name for path in box.old_copy.parent.iterdir()] == ["page-check"] and not box.staging.exists()


def test_the_copy_creates_the_folders_when_the_browser_container_has_none(box):
    cached(box)
    source = tree(box.source_folder)
    shutil.rmtree(box.fs / box.browser_c)
    (box.fs / box.browser_c).mkdir()
    expect_success(box.run())
    assert tree(box.old_copy) == source


def test_renamed_containers_of_the_template_are_the_ones_used(box):
    cached(box)
    box.edit_template("raphael-journeys-app", "owner-app-renamed")
    box.edit_template("raphael-journeys-browser", "browser-renamed")
    source = tree(box.source_folder)
    expect_success(box.run())
    assert box.kinds() == CACHED_RUN
    (command,) = box.remote("copy-page-check")
    (publish,) = box.remote("publish-page-check")
    assert "'owner-app-renamed:" in command and "'browser-renamed:" in command and "raphael-journeys-" not in command
    assert publish.startswith("docker exec 'browser-renamed' sh -c ") and "raphael-journeys-" not in publish
    (sweep,) = box.remote("rm-page-check")
    assert sweep.startswith("docker exec 'browser-renamed' sh -c ") and "raphael-journeys-" not in sweep
    assert tree(box.old_copy) == source and "browser-renamed" in str(box.old_copy)


def _no_source(box):
    shutil.rmtree(box.source_folder)


def _empty_source(box):
    for path in list(box.source_folder.rglob("*"))[::-1]:
        path.unlink() if path.is_file() else path.rmdir()


def _no_browser(box):
    shutil.rmtree(box.fs / box.browser_c)


STAGE, PUBLISH = "copy-page-check", "publish-page-check"
PAGE_FAILS = {
    "no source": (_no_source, STAGE), "empty source": (_empty_source, STAGE), "no browser": (_no_browser, STAGE),
    "producer fails with 23 after a valid partial archive": (lambda box: box.answer(rc={"s2docker cp-out": 23}), STAGE),
    "consumer fails": (lambda box: box.answer(rc={"s2docker cp-in": 1}), STAGE),
    "remote shell fails after the whole stage ran": (lambda box: box.answer(late_rc={STAGE: 255}), STAGE),
    "swap fails and rolls back": (lambda box: (box.top / "mv-fails").write_text("x", encoding="utf-8"), PUBLISH),
    "forced failure of the stage": (lambda box: box.answer(rc={STAGE: 1}), STAGE),
    "forced failure of the publish": (lambda box: box.answer(rc={PUBLISH: 1}), PUBLISH),
}


@pytest.mark.parametrize("setup, last", list(PAGE_FAILS.values()), ids=list(PAGE_FAILS))
def test_a_failed_copy_ends_the_run_keeps_the_old_copy_untouched_and_removes_nothing(box, setup, last):
    cached(box)
    box.answer(tags=RETENTION_TAGS)
    setup(box)
    before = box.server2()
    expect_failure(box.run(), "page-check")
    if (box.fs / box.browser_c).exists():
        assert tree(box.old_copy) == {"old.txt": "OLD\n"}
    assert box.server2() == after_converge(before)
    assert box.kinds()[-1] == last and box.of("remove-tag") == [] and box.of("list-tags") == []


# ---- owner rule 2: stage, check every side, then publish. One regression per transfer ---------------

SCRIPTS_STAGE = [*GATE, "ship-scripts"]
APP_IMAGE_STAGE = [*CACHED_RUN[:4], "load"]
PLATFORM_STAGE = [*CACHED_RUN[:5], "ship-platform"]
PLATFORM_RUN = [*PLATFORM_STAGE, "publish-platform", "build-platform", "has-image", *CACHED_RUN[5:]]
PAGE_STAGE = CACHED_RUN[:CACHED_RUN.index("copy-page-check") + 1]
OLD_STACK_UP = "#!/bin/sh\n# old stack_up\n"


def old_scripts(box: Sandbox) -> None:
    """Server 2 holds an older scripts folder; its marker names another commit, so the scripts must ship."""
    box.set_marker(OTHER_COMMIT + "\n")
    (box.s2dir / "old.txt").write_text("OLD\n", encoding="utf-8")
    (box.s2dir / "stack_up.sh").write_text(OLD_STACK_UP, encoding="utf-8")


def no_images(box: Sandbox) -> None:
    """Server 2 holds the scripts of this commit and no image at all."""
    box.set_marker(COMMIT + "\n")


def tagged_app_image(box: Sandbox) -> None:
    """Server 2 holds the scripts of this commit and the owner app image with its tag, but no platform image."""
    no_platform(box)
    box.answer(images=[IMAGE_ID, APP_BASE_TAG])


def after_converge(before: dict) -> dict:
    """The steps before the page check do change the fake Server 2: the app image is tagged and the stack converged."""
    return {**before, "converged": True, "refs": sorted({*before["refs"], APP_BASE_TAG})}


def fails_unpublished(box: Sandbox, key: str, expected: dict, kinds: list[str]) -> None:
    """Runs the script: it ends with the fixed reason of its transfer after exactly these calls, and Server 2 holds
    exactly `expected` (what it held before, plus what only earlier steps of the run change)."""
    expect_failure(box.run(), key)
    assert box.kinds() == kinds
    assert box.server2() == expected


COMPLETE_THEN_23 = {"late_rc": {"git archive scripts": 23}}
CUT_THEN_23 = {"late_rc": {"git archive scripts": 23}, "cut": {"git archive scripts": "stack_up.sh"}}
PRODUCER_IDS = ["whole stream", "tar cut before stack_up.sh"]


@pytest.mark.parametrize("answers, missing", [(COMPLETE_THEN_23, set()), (CUT_THEN_23, {"stack_up.sh"})], ids=PRODUCER_IDS)
def test_a_scripts_producer_that_exits_non_zero_after_its_stream_leaves_the_folder_and_marker_as_they_were(box, answers, missing):
    old_scripts(box)
    before = box.server2()
    box.answer(**answers)
    fails_unpublished(box, "ship-scripts", before, SCRIPTS_STAGE)
    assert tree(box.s2dir) == {"old.txt": "OLD\n", "stack_up.sh": OLD_STACK_UP, ".refresh-commit": OTHER_COMMIT + "\n"}
    assert before["files"]["raphael-journeys/.refresh-commit"] == (OTHER_COMMIT + "\n").encode("utf-8")
    (git,), (ship,) = box.of("git archive scripts"), box.of("ship-scripts")
    assert ship["stdin_sha"] == git["sha"], "the receiver was given the stream the producer wrote"
    staged = tree(Path(f"{box.s2dir}.new"))
    assert staged == {n: t for n, t in expected_scripts(box).items() if n not in missing}, "real tar took the stream"


@pytest.mark.parametrize("answers", [COMPLETE_THEN_23, CUT_THEN_23], ids=PRODUCER_IDS)
def test_a_healthy_retry_at_the_same_commit_ships_the_scripts_again_and_recovers(box, answers):
    old_scripts(box)
    before = box.server2()
    box.answer(**answers)
    fails_unpublished(box, "ship-scripts", before, SCRIPTS_STAGE)
    box.answer(late_rc={"git archive scripts": 0}, cut={"git archive scripts": None})
    expect_success(box.run())
    assert box.kinds() == SCRIPTS_STAGE + FULL_RUN
    archives, ships = box.of("git archive scripts"), box.of("ship-scripts")
    assert len(archives) == len(ships) == 2 and ships[1]["stdin_sha"] == archives[1]["sha"] and ships[1]["stdin_size"] > 0
    assert tree(box.s2dir) == {**expected_scripts(box), ".refresh-commit": COMMIT + "\n"}
    previous = {"old.txt": "OLD\n", "stack_up.sh": OLD_STACK_UP, ".refresh-commit": OTHER_COMMIT + "\n"}
    assert tree(box.s2dir.parent / "raphael-journeys.previous") == previous


def test_a_platform_producer_that_exits_non_zero_after_its_stream_publishes_no_archive_and_never_builds(box):
    tagged_app_image(box)
    before = box.server2()
    box.answer(late_rc={"git archive commit": 23})
    fails_unpublished(box, "platform-build", before, PLATFORM_STAGE)
    assert not Path(box.archive()).exists() and before["built"] is None and box.of("build-platform") == []
    (git,), (ship,) = box.of("git archive commit"), box.of("ship-platform")
    assert ship["stdin_sha"] == git["sha"], "the receiver was given the stream the producer wrote"


def test_a_healthy_retry_at_the_same_commit_ships_the_platform_archive_again_and_goes_on(box):
    tagged_app_image(box)
    before = box.server2()
    box.answer(late_rc={"git archive commit": 23})
    fails_unpublished(box, "platform-build", before, PLATFORM_STAGE)
    box.answer(late_rc={"git archive commit": 0})
    expect_success(box.run())
    assert box.kinds() == PLATFORM_STAGE + PLATFORM_RUN
    archives, ships = box.of("git archive commit"), box.of("ship-platform")
    assert len(archives) == len(ships) == 2 and ships[1]["stdin_sha"] == archives[1]["sha"]
    state = json.loads((box.top / "state.json").read_text(encoding="utf-8"))
    assert state["archive_sha"] == archives[1]["sha"] and PLATFORM_REF in state["images"]
    assert not Path(box.archive()).exists(), "the final reconciliation removes the archive once the stack is current"
    assert not Path(f"{box.archive()}.part").exists() and reconciled(box)


def test_an_archive_that_is_already_gone_when_it_is_removed_is_no_failure(box):
    no_platform(box)
    box.answer(eats_archive=True)
    expect_success(box.run())
    assert box.kinds() == PLATFORM_RUN and not Path(box.archive()).exists()


def test_an_app_image_producer_that_exits_non_zero_after_its_stream_makes_no_tag_and_stops_the_run(box):
    no_images(box)
    before = box.server2()
    box.answer(late_rc={"docker save": 23})
    fails_unpublished(box, "app-image", before, APP_IMAGE_STAGE)
    assert box.of("tag") == [] and APP_BASE_TAG not in box.server2()["refs"] and before["refs"] == []
    (save,), (load,) = box.of("docker save"), box.of("load")
    assert load["stdin_sha"] == save["sha"], "the receiver was given the stream the producer wrote"


def test_a_retry_after_a_load_and_a_failed_producer_finds_the_id_without_a_tag_and_ships_nothing_again(box):
    no_images(box)
    before = box.server2()
    box.answer(late_rc={"docker save": 23})
    fails_unpublished(box, "app-image", before, APP_IMAGE_STAGE)
    state = json.loads((box.top / "state.json").read_text(encoding="utf-8"))
    assert box.of("tag") == [] and state["untagged"] == [IMAGE_ID], "the loaded id waits on Server 2 without a tag"
    box.answer(late_rc={"docker save": 0})
    expect_success(box.run())
    assert len(box.of("docker save")) == len(box.of("load")) == 1, "the retry shipped the image again"
    assert box.kinds() == APP_IMAGE_STAGE + PLATFORM_RUN
    assert box.remote("tag") == [f"docker tag '{IMAGE_ID}' 'raphael-journeys-app-base:{REVISION}'"]
    assert APP_BASE_TAG in box.server2()["refs"]


def test_a_page_check_producer_that_exits_non_zero_after_a_valid_stream_swaps_nothing_and_stops_before_the_retention(box):
    cached(box)
    box.answer(tags=RETENTION_TAGS)
    before = box.server2()
    box.answer(late_rc={"s2docker cp-out": 23})
    fails_unpublished(box, "page-check", after_converge(before), PAGE_STAGE)
    assert before["page"] == {"old.txt": "OLD\n"} and tree(box.old_copy) == before["page"]
    assert [c["call"] for c in box.calls() if c["call"].startswith("s2docker")] == ["s2docker exec", "s2docker cp-out"]
    assert box.kinds().count("versions") == 1 and box.of("list-tags") == [] and box.of("remove-tag") == []


SIDE_FAILS = {
    "scripts, the receiver exits 1": (old_scripts, {"rc": {"ship-scripts": 1}}, "ship-scripts", SCRIPTS_STAGE),
    "scripts, the receiver exits 255 after it extracted everything": (
        old_scripts, {"late_rc": {"ship-scripts": 255}}, "ship-scripts", SCRIPTS_STAGE),
    "scripts, the publish call fails": (
        old_scripts, {"rc": {"publish-scripts": 1}}, "ship-scripts", [*SCRIPTS_STAGE, "publish-scripts"]),
    "platform, the receiver exits 1": (tagged_app_image, {"rc": {"ship-platform": 1}}, "platform-build", PLATFORM_STAGE),
    "platform, the receiver exits 255 after it wrote the whole file": (
        tagged_app_image, {"late_rc": {"ship-platform": 255}}, "platform-build", PLATFORM_STAGE),
    "platform, the publish call fails": (
        tagged_app_image, {"rc": {"publish-platform": 1}}, "platform-build", [*PLATFORM_STAGE, "publish-platform"]),
    "app image, the receiver exits 1": (no_images, {"rc": {"load": 1}}, "app-image", APP_IMAGE_STAGE),
    "app image, the receiver exits 255 after it loaded the image": (
        no_images, {"late_rc": {"load": 255}}, "app-image", APP_IMAGE_STAGE),
    "app image, the publish call fails": (no_images, {"rc": {"tag": 1}}, "app-image", [*APP_IMAGE_STAGE, "tag"]),
}


@pytest.mark.parametrize("seed, answers, key, kinds", list(SIDE_FAILS.values()), ids=list(SIDE_FAILS))
def test_a_receiver_or_publish_call_that_fails_leaves_the_active_state_as_it_was_and_ends_with_its_reason(
        box, seed, answers, key, kinds):
    seed(box)
    before = box.server2()
    box.answer(**answers)
    fails_unpublished(box, key, before, kinds)


def test_a_failed_publish_of_the_scripts_leaves_the_staged_folder_without_the_marker(box):
    old_scripts(box)
    box.answer(rc={"publish-scripts": 1})
    expect_failure(box.run(), "ship-scripts")
    assert tree(Path(f"{box.s2dir}.new")) == expected_scripts(box), "only the publish call writes the marker"
    assert tree(box.s2dir)[".refresh-commit"] == OTHER_COMMIT + "\n"


PUBLISH_FAILS = {
    "the first rename": ("mv-fails-first", True, False),
    "the second rename": ("mv-fails-second", True, False),
    "the removal of a stale previous folder": ("rm-fails-previous", True, True),
    "the only rename of a first run": ("mv-fails-second", False, False),
}


@pytest.mark.parametrize("control, has_old, has_stale", list(PUBLISH_FAILS.values()), ids=list(PUBLISH_FAILS))
def test_a_scripts_publish_that_fails_at_any_step_keeps_the_old_folder_and_marker_and_a_retry_recovers(
        box, control, has_old, has_stale):
    previous = Path(f"{box.s2dir}.previous")
    if has_old:
        old_scripts(box)
    if has_stale:
        previous.mkdir()
        (previous / "stale.txt").write_text("STALE\n", encoding="utf-8")
    kept, stale = tree(box.s2dir) if has_old else None, tree(previous) if has_stale else None
    before = box.server2()
    (box.top / control).touch()
    fails_unpublished(box, "ship-scripts", before, [*SCRIPTS_STAGE, "publish-scripts"])
    if has_old:
        assert tree(box.s2dir) == kept and kept[".refresh-commit"] == OTHER_COMMIT + "\n"
    else:
        assert not box.s2dir.exists()
    assert (tree(previous) if previous.exists() else None) == stale
    (box.top / control).unlink()
    expect_success(box.run())
    assert box.kinds() == [*SCRIPTS_STAGE, "publish-scripts", *FULL_RUN]
    assert tree(box.s2dir) == {**expected_scripts(box), ".refresh-commit": COMMIT + "\n"}
    assert (tree(previous) if previous.exists() else None) == kept


def test_a_failed_publish_of_the_platform_archive_leaves_the_staged_file_and_no_archive_under_the_final_name(box):
    tagged_app_image(box)
    box.answer(rc={"publish-platform": 1})
    expect_failure(box.run(), "platform-build")
    assert Path(f"{box.archive()}.part").is_file() and not Path(box.archive()).exists()
    assert box.of("build-platform") == []


# ---- g. the retention: last, with its own versions call, by exact references ------------------------

LIST_FORMAT = "--format '{{.Tag}}'"
PLATFORM_REPO = "raphael-journeys-platform"
BAD_TAGS = ["<none>", "bad tag!", "x" * 64]


def retention_steps(box: Sandbox) -> list[tuple[str, str]]:
    """The ssh calls after the last versions call, as (kind, repository or reference)."""
    calls = box.ssh_calls()
    start = max(i for i, call in enumerate(calls) if call["call"] == "versions") + 1
    return [(call["call"], call["arg"]) for call in calls[start:]]


def planned(removed: dict[str, list[str]]) -> list[tuple[str, str]]:
    steps: list[tuple[str, str]] = []
    for repo in REPOSITORIES:
        steps.append(("list-tags", repo))
        steps += [("remove-tag", f"{repo}:{tag}") for tag in removed[repo]]
    return steps


def stored_tags(box: Sandbox) -> dict[str, list[str]]:
    return json.loads((box.top / "state.json").read_text(encoding="utf-8"))["tags"]


def test_the_retention_is_last_and_removes_every_old_plain_tag_by_its_exact_reference_one_call_each(shared):
    box, done = shared("tags")
    expect_success(done)
    assert box.kinds()[:8] == CURRENT_RUN[:8] and reconciled(box), "the reconciliation comes before the retention"
    assert box.remote("versions")[-1] == box.cmd("./stack_up.sh versions")
    assert retention_steps(box) == planned(REMOVED)
    assert box.remote("list-tags") == [f"docker image ls '{repo}' {LIST_FORMAT}" for repo in REPOSITORIES]
    assert box.remote("remove-tag") == [f"docker image rm '{repo}:{tag}'" for repo in REPOSITORIES for tag in REMOVED[repo]]
    assert all(call["stdin_size"] == 0 for call in box.ssh_calls())
    assert stored_tags(box) == {r: [t for t in RETENTION_TAGS[r] if t not in REMOVED[r]] for r in REPOSITORIES}


def test_none_and_non_plain_tags_are_neither_counted_nor_removed(shared):
    box, _ = shared("tags")
    assert stored_tags(box)[PLATFORM_REPO] == [*BAD_TAGS[:2], "p1" * 20, "p2" * 20, BAD_TAGS[2], COMMIT]
    assert not [call["arg"] for call in box.of("remove-tag") if call["arg"].split(":", 1)[1] in BAD_TAGS]


def test_tags_used_by_containers_are_those_of_the_retention_call_and_per_repository(box):
    with_tags(box)
    tags = {**RETENTION_TAGS, "raphael-journeys-browser": [f"{OLD[1]}-{CA}", f"{OLD[2]}-{CA}", APP_TAG]}
    box.answer(tags=tags, out={"versions#2": versions_text(
        proxy="c" * 12, app=APP_TAG, dashboard="p3" * 20, gateway="p4" * 20, browser="none")})
    expect_success(box.run())
    removed = {
        "raphael-journeys-proxy": [PROXY_TAG, "d" * 12], "raphael-journeys-app-base": REMOVED["raphael-journeys-app-base"],
        "raphael-journeys-app": [f"{OLD[3]}-{CA}", f"{REVISION}-{'b' * 12}", f"{OLD[4]}-{CA}"],
        PLATFORM_REPO: [COMMIT], "raphael-journeys-browser": [APP_TAG],
    }
    assert retention_steps(box) == planned(removed)


BAD_RETENTION_VERSIONS = {
    "fails": (versions_text(), 1), "ssh fails": (versions_text(), 255), "garbled": ("garbage\n", 0), "empty": ("", 0),
    "four lines": (versions_text().rsplit("browser", 1)[0], 0), "six lines": (versions_text() + "extra x\n", 0),
    "wrong order": (_swapped(), 0), "blank line": (versions_text().replace("app ", "\napp "), 0),
    "tag not plain": (versions_text(proxy="a!b"), 0), "tag with a slash": (versions_text(app="a/b"), 0),
    "tag too long": (versions_text(browser="a" * 64), 0), "empty tag": (versions_text(gateway=""), 0),
    "unknown role": (versions_text().replace("gateway", "gatewa"), 0),
    "extra field": (versions_text().replace("proxy ", "proxy x "), 0),
}


@pytest.mark.parametrize("text, rc", list(BAD_RETENTION_VERSIONS.values()), ids=list(BAD_RETENTION_VERSIONS))
def test_versions_that_cannot_be_read_at_the_retention_remove_nothing_and_end_with_prune(box, text, rc):
    with_tags(box)
    box.answer(out={"versions#2": text}, rc={"versions#2": rc})
    expect_failure(box.run(), "prune")
    assert box.kinds() == CURRENT_RUN[:8] and stored_tags(box) == RETENTION_TAGS


@pytest.mark.parametrize("target, listed", [
    (f"raphael-journeys-proxy:{'c' * 12}", 1), (f"{PLATFORM_REPO}:{'p4' * 20}", 4),
], ids=["first removal", "last removal"])
def test_a_removal_that_fails_ends_with_prune_after_the_converge_and_the_page_check_ran(box, target, listed):
    cached(box)
    box.answer(tags=RETENTION_TAGS, rc={f"remove-tag {target}": 1})
    expect_failure(box.run(), "prune")
    kinds = box.kinds()
    assert kinds[-1] == "remove-tag" and "converge" in kinds and kinds.index("copy-page-check") < kinds.index("list-tags")
    assert box.of("remove-tag")[-1]["arg"] == target and [c["arg"] for c in box.of("list-tags")] == REPOSITORIES[:listed]
    assert tree(box.old_copy) == tree(box.source_folder)


def test_a_list_that_fails_ends_with_prune_and_no_later_repository_is_touched(box):
    cached(box)
    box.answer(tags=RETENTION_TAGS, rc={"list-tags raphael-journeys-app": 1})
    expect_failure(box.run(), "prune")
    assert box.kinds()[-1] == "list-tags" and [c["arg"] for c in box.of("list-tags")] == REPOSITORIES[:3]
    assert [c["arg"] for c in box.of("remove-tag")] == [f"{r}:{t}" for r in REPOSITORIES[:2] for t in REMOVED[r]]


def test_a_full_run_ends_with_the_retention_and_removes_nothing_from_a_stack_that_has_two_tags_or_fewer(shared):
    box, done = shared("full")
    expect_success(done)
    assert retention_steps(box) == planned({repo: [] for repo in REPOSITORIES})
    assert box.of("remove-tag") == []


# ---- owner rule 3: one final reconciliation. It removes by exact name, once the stack is current -----------------
# A failed removal is remembered: convergence, the page check copy and the retention still finish, and the run ends
# with the fixed reason of the first failed phase. The next healthy run removes what is left.

LEFTOVERS = sorted([
    "raphael-journeys.new", f"raphael-journeys.platform-{COMMIT}.tar", f"raphael-journeys.platform-{COMMIT}.tar.part",
    "raphael-journeys.page-check.tar", "page-check.new", "page-check.old",
])
PLAIN_REMAINS = {repo: [t for t in RETENTION_TAGS[repo] if t not in REMOVED[repo]] for repo in REPOSITORIES}
CLEANUPS = {
    "the platform archive": (no_platform, "rm-archive", "rm-fails-archive", "platform-build", lambda box: Path(box.archive())),
    "the old page check": (cached, "rm-page-check", "rm-fails", "page-check", lambda box: Path(f"{box.old_copy}.old")),
    "the staging tar of the page check": (
        cached, "rm-staging-tar", "rm-fails-staging", "page-check", lambda box: box.staging),
    "the staged scripts folder": (
        cached, "rm-scripts-new", "rm-fails-new", "ship-scripts", lambda box: Path(f"{box.s2dir}.new")),
}
INJECTIONS = ["forced status", "the tool of Server 2 fails"]


def inject(box: Sandbox, how: str, kind: str, control: str, on: bool) -> None:
    if how == "forced status":
        box.answer(rc={kind: 255 if on else None})
    elif on:
        (box.top / control).touch()
    else:
        (box.top / control).unlink()


def stack_is_up_but_not_clean(box: Sandbox, leftover: Path) -> None:
    """After a run that ended on a failed removal: the stack was brought up, the page check is the new one and the
    retention ran through; only the one leftover is there."""
    assert len(box.of("build-proxy")) == len(box.of("converge")) == 1 and box.server2()["converged"] is True
    assert tree(box.old_copy) == tree(box.source_folder) and stored_tags(box) == PLAIN_REMAINS
    assert retention_steps(box) == planned(REMOVED) and leftover.exists()


@pytest.mark.parametrize("how", INJECTIONS)
@pytest.mark.parametrize("what", list(CLEANUPS), ids=list(CLEANUPS))
def test_a_removal_that_fails_ends_with_the_reason_of_its_phase_after_the_stack_is_current_and_the_next_run_removes_the_leftover(
        box, what, how):
    scenario, kind, control, reason, where = CLEANUPS[what]
    scenario(box)
    box.answer(tags=RETENTION_TAGS)
    if kind == "rm-scripts-new":
        box.plant_leftovers()
    inject(box, how, kind, control, True)
    expect_failure(box.run(), reason)
    kinds = box.kinds()
    assert kinds.index("converge") < kinds.index("copy-page-check") < kinds.index(kind) < kinds.index("list-tags")
    assert [k for k in kinds if k in RECONCILE] == RECONCILE, "every removal was tried although one failed"
    stack_is_up_but_not_clean(box, where(box))
    inject(box, how, kind, control, False)
    expect_success(box.run())
    assert box.last_kinds() == CURRENT_RUN, "the second run takes the already-current shortcut"
    assert box.leftovers() == [] and not where(box).exists()
    assert len(box.of("build-proxy")) == len(box.of("converge")) == 1 and stored_tags(box) == PLAIN_REMAINS


@pytest.mark.parametrize("how", INJECTIONS)
def test_both_page_check_leftovers_that_could_not_be_removed_are_removed_together_by_the_next_run_through_the_shortcut(
        box, how):
    cached(box)
    box.answer(tags=RETENTION_TAGS)
    both = (("rm-page-check", "rm-fails"), ("rm-staging-tar", "rm-fails-staging"))
    for kind, control in both:
        inject(box, how, kind, control, True)
    expect_failure(box.run(), "page-check")
    assert box.leftovers() == ["page-check.old", "raphael-journeys.page-check.tar"]
    assert tree(box.old_copy) == tree(box.source_folder) and stored_tags(box) == PLAIN_REMAINS
    for kind, control in both:
        inject(box, how, kind, control, False)
    expect_success(box.run())
    assert box.last_kinds() == CURRENT_RUN and box.leftovers() == []
    assert sorted(path.name for path in box.old_copy.parent.iterdir()) == ["page-check"] and not box.staging.exists()


@pytest.mark.parametrize("target, remaining", [
    (f"raphael-journeys-proxy:{'c' * 12}", planned(REMOVED)),
    (f"{PLATFORM_REPO}:{'p4' * 20}", planned({**{repo: [] for repo in REPOSITORIES}, PLATFORM_REPO: ["p4" * 20]})),
], ids=["first removal", "last removal"])
def test_a_tag_that_cannot_be_removed_ends_with_prune_after_the_stack_is_current_and_the_next_run_removes_it(
        box, target, remaining):
    cached(box)
    box.answer(tags=RETENTION_TAGS, rc={f"remove-tag {target}": 1})
    expect_failure(box.run(), "prune")
    kinds = box.kinds()
    assert "converge" in kinds and kinds.index("publish-page-check") < kinds.index("list-tags") and reconciled(box)
    assert box.of("remove-tag")[-1]["arg"] == target and tree(box.old_copy) == tree(box.source_folder)
    assert target.split(":")[1] in stored_tags(box)[target.split(":")[0]], "the tag is still there"
    box.answer(rc={f"remove-tag {target}": None})
    expect_success(box.run())
    assert box.last_kinds()[:3] == CURRENT_RUN[:3] and retention_steps(box) == remaining
    assert stored_tags(box) == PLAIN_REMAINS and len(box.of("converge")) == 1


FIRST_FAILED = {
    "archive, then a tag": ({"rm-archive": 1, "remove-tag raphael-journeys-proxy:" + "c" * 12: 1}, "platform-build"),
    "old page check, then a tag": ({"rm-page-check": 1, "remove-tag raphael-journeys-proxy:" + "c" * 12: 1}, "page-check"),
    "staging tar, then a tag": ({"rm-staging-tar": 255, "remove-tag raphael-journeys-proxy:" + "c" * 12: 1}, "page-check"),
    "archive, then the old page check": ({"rm-archive": 255, "rm-page-check": 1}, "platform-build"),
    "old page check, then the staging tar": ({"rm-page-check": 1, "rm-staging-tar": 1}, "page-check"),
    "staged scripts, then the archive": ({"rm-scripts-new": 1, "rm-archive": 1}, "ship-scripts"),
    "all four removals and a tag": ({
        "rm-scripts-new": 1, "rm-archive": 1, "rm-page-check": 1, "rm-staging-tar": 1,
        "remove-tag raphael-journeys-proxy:" + "c" * 12: 1}, "ship-scripts"),
}


@pytest.mark.parametrize("rc, reason", list(FIRST_FAILED.values()), ids=list(FIRST_FAILED))
def test_two_failed_cleanups_end_with_the_reason_of_the_first_failed_phase_and_the_work_is_finished(box, rc, reason):
    no_platform(box)
    box.answer(tags=RETENTION_TAGS, rc=rc)
    expect_failure(box.run(), reason)
    kinds = box.kinds()
    assert kinds.count("build-proxy") == kinds.count("converge") == kinds.count("publish-page-check") == 1
    assert [k for k in kinds if k in RECONCILE] == RECONCILE and "versions" in kinds[kinds.index("rm-staging-tar"):]
    assert tree(box.old_copy) == tree(box.source_folder)
    box.answer(rc={key: None for key in rc})
    expect_success(box.run())
    assert box.leftovers() == [] and stored_tags(box) == PLAIN_REMAINS


@pytest.mark.parametrize("key", [
    "rm-archive", "rm-page-check", "rm-staging-tar", "versions#2", "list-tags raphael-journeys-proxy", "remove-tag#1",
])
def test_the_box_running_out_after_a_failed_cleanup_is_the_box_and_not_the_remembered_reason(box, key):
    no_platform(box)
    box.answer(tags=RETENTION_TAGS, rc={"rm-scripts-new": 255}, advance={key: BOX})
    expect_failure(box.run(), "box")
    kinds = box.kinds()
    assert kinds[-1] == key.split()[0].split("#")[0] and box.elapsed() == BOX and "rm-scripts-new" in kinds
    assert kinds.count("converge") == 1
    nothing_stopped(box)


def test_the_previous_scripts_copy_survives_every_reconciliation(box):
    old_scripts(box)
    box.answer(images=[IMAGE_ID, PLATFORM_REF], tags=RETENTION_TAGS, rc={"rm-archive": 255})
    old = tree(box.s2dir)
    previous = Path(f"{box.s2dir}.previous")
    expect_failure(box.run(), "platform-build")
    assert tree(previous) == old and reconciled(box)
    box.answer(rc={"rm-archive": None})
    expect_success(box.run())
    assert tree(previous) == old and box.last_kinds() == CURRENT_RUN
    expect_success(box.run())
    assert tree(previous) == old and box.leftovers() == []
    sweeps = [cmd for kind in RECONCILE for cmd in box.remote(kind)]
    assert len(sweeps) == 12 and not any(".previous" in cmd or ".refresh-commit" in cmd for cmd in sweeps)
    assert not any(re.search(r"[*?]|\bdocker (rm|container|volume)\b", cmd) for cmd in sweeps)
    assert (box.s2dir / ".refresh-commit").read_text(encoding="utf-8") == COMMIT + "\n"


@pytest.mark.parametrize("scenario", ["cached", "current"])
def test_a_healthy_run_removes_every_leftover_of_failed_runs_by_exact_name_and_nothing_that_only_looks_like_one(
        box, scenario):
    SCENARIOS[scenario](box)
    previous = Path(f"{box.s2dir}.previous")
    previous.mkdir()
    (previous / "kept.txt").write_text("KEPT\n", encoding="utf-8")
    (box.s2dir / "stack_up.sh").write_text(OLD_STACK_UP, encoding="utf-8")
    assert box.plant_leftovers() == LEFTOVERS
    decoys = [
        Path(box.archive(OTHER_COMMIT)), Path(f"{box.s2dir}.newer"), Path(f"{box.s2dir}.page-check.tar.bak"),
        Path(f"{box.s2dir}.platform-{COMMIT}.tar.part2"), box.old_copy.parent / "page-check.older",
    ]
    for decoy in decoys:
        if decoy.parent == box.old_copy.parent:
            decoy.mkdir()
        else:
            decoy.write_bytes(b"DECOY")
    expect_success(box.run())
    assert box.leftovers() == sorted(d.name for d in decoys) and all(d.exists() for d in decoys)
    assert reconciled(box) and tree(previous) == {"kept.txt": "KEPT\n"}
    assert (box.s2dir / "stack_up.sh").read_text(encoding="utf-8") == OLD_STACK_UP
    assert (box.s2dir / ".refresh-commit").read_text(encoding="utf-8") == COMMIT + "\n"
    assert box.kinds() == (CURRENT_RUN if scenario == "current" else CACHED_RUN)


STAGED = {
    "scripts publish fails": (old_scripts, {"rc": {"publish-scripts": 1}}, "ship-scripts", ["raphael-journeys.new"]),
    "platform publish fails": (
        tagged_app_image, {"rc": {"publish-platform": 1}}, "platform-build", [f"raphael-journeys.platform-{COMMIT}.tar.part"]),
    "page check consumer fails": (
        cached, {"rc": {"s2docker cp-in": 1}}, "page-check", ["page-check.new", "raphael-journeys.page-check.tar"]),
}


@pytest.mark.parametrize("seed, answers, reason, left", list(STAGED.values()), ids=list(STAGED))
def test_the_staged_names_a_failed_run_leaves_are_tolerated_and_cleaned_by_the_next_healthy_run(
        box, seed, answers, reason, left):
    seed(box)
    box.answer(**answers)
    expect_failure(box.run(), reason)
    assert box.leftovers() == sorted(left)
    box.answer(**{what: {key: None for key in value} for what, value in answers.items()})
    expect_success(box.run())
    assert box.leftovers() == [] and box.server2()["converged"] is True


def test_a_failed_swap_of_the_page_check_leaves_the_old_copy_active_and_the_next_run_cleans_the_new_sibling(box):
    cached(box)
    (box.top / "mv-fails").write_text("x", encoding="utf-8")
    expect_failure(box.run(), "page-check")
    assert tree(box.old_copy) == {"old.txt": "OLD\n"}
    assert box.leftovers() == ["page-check.new", "raphael-journeys.page-check.tar"]
    assert box.of("list-tags") == [] and "rm-page-check" not in box.kinds(), "a failed copy reconciles and prunes nothing"
    (box.top / "mv-fails").unlink()
    expect_success(box.run())
    assert box.leftovers() == [] and box.old_copy.is_dir()


def run_on_server2(box: Sandbox, command: str) -> subprocess.CompletedProcess[str]:
    """One recorded remote command, run for real in the folders of the fake Server 2 with the same tools as the stand-in."""
    env = {"PATH": str(box.s2bin), "HOME": str(box.top)}
    return subprocess.run(["/bin/sh", "-c", command], env=env, capture_output=True, encoding="utf-8", text=True, cwd=box.top)


def swap_ready(box: Sandbox) -> tuple[str, Path, Path]:
    """The swap command of a normal run, with Server 2 put back to: the old copy active, a new copy beside it staged, and
    an old sibling of an earlier failed run in the way."""
    cached(box)
    expect_success(box.run())
    (swap,) = box.remote("publish-page-check")
    box.seed_fs()
    new, old = Path(f"{box.old_copy}.new"), Path(f"{box.old_copy}.old")
    shutil.copytree(box.source_folder, new)
    (old / "stale").mkdir(parents=True)
    (old / "stale" / "file.txt").write_text("STALE\n", encoding="utf-8")
    return swap, new, old


def test_a_page_check_swap_that_meets_an_old_sibling_removes_that_exact_leftover_first(box):
    swap, new, old = swap_ready(box)
    done = run_on_server2(box, swap)
    assert done.returncode == 0, done.stderr
    assert tree(box.old_copy) == tree(box.source_folder) and not new.exists()
    assert tree(old) == {"old.txt": "OLD\n"}, "the active copy was moved to the exact .old name, not into a leftover"


def test_a_page_check_swap_that_cannot_remove_the_old_sibling_moves_nothing(box):
    swap, new, old = swap_ready(box)
    (box.top / "rm-fails").write_text("x", encoding="utf-8")
    done = run_on_server2(box, swap)
    assert done.returncode == 1
    assert tree(box.old_copy) == {"old.txt": "OLD\n"} and tree(new) == tree(box.source_folder)
    assert tree(old) == {"stale/file.txt": "STALE\n"}


# ---- A. one deadline of 25 minutes for the whole run -----------------------------------------------

AFTER_BOX = {
    "build-proxy", "converge", "copy-page-check", "publish-page-check", *RECONCILE, "list-tags", "remove-tag",
}


def nothing_stopped(box: Sandbox) -> None:
    """On the box nothing on Server 2 is killed, stopped or taken down."""
    for call in box.ssh_calls():
        assert not re.search(r"\b(kill|pkill|stop|restart|down)\b", call["remote"]), call["remote"]


def test_a_platform_build_call_that_returns_late_is_the_box_even_when_the_image_is_there_at_once(box):
    no_platform(box)
    box.answer(advance={"build-platform": BOX + 1})
    expect_failure(box.run(), "box")
    kinds = box.kinds()
    assert kinds[-1] == "build-platform" and kinds.count("build-platform") == 1 and not AFTER_BOX & set(kinds)
    nothing_stopped(box)


def test_a_converge_that_returns_late_is_the_box_and_no_page_check_or_removal_follows(box):
    cached(box)
    box.answer(tags=RETENTION_TAGS, advance={"converge": BOX + 1})
    expect_failure(box.run(), "box")
    kinds = box.kinds()
    assert kinds[-1] == "converge" and kinds.count("versions") == 1
    assert not {"copy-page-check", "publish-page-check", *RECONCILE, "list-tags", "remove-tag"} & set(kinds)
    assert tree(box.old_copy) == {"old.txt": "OLD\n"}


def test_a_first_look_that_returns_late_is_the_box_and_not_success(box):
    current(box)
    box.answer(advance={"versions#1": BOX + 1})
    expect_failure(box.run(), "box")
    assert box.kinds() == ["marker", "versions"]


@pytest.mark.parametrize("late", [BOX, BOX + 1, BOX + 600])
def test_a_clock_past_the_box_when_the_shipping_would_begin_starts_no_shipping_call(box, late):
    box.answer(advance={"marker": late})
    expect_failure(box.run(), "box")
    assert box.kinds() == ["marker"] and box.server1() == ["git toplevel", "git head", "docker inspect"]


@pytest.mark.parametrize("key, stream, later", [
    ("has-image#1", "docker save", {"load", "tag", "ship-platform", "publish-platform", "build-platform"}),
    ("has-image#2", "git archive commit", {"ship-platform", "publish-platform", "build-platform"}),
], ids=["app image", "platform image"])
def test_a_clock_past_the_box_after_a_question_starts_no_stream(box, key, stream, later):
    box.answer(advance={key: BOX})
    expect_failure(box.run(), "box")
    assert stream not in box.server1() and not later & set(box.kinds())


@pytest.mark.parametrize("label", ["git toplevel", "git head", "docker inspect"])
def test_a_call_on_server_1_that_returns_late_is_the_box_and_nothing_follows(box, label):
    box.answer(advance={label: BOX + 1})
    expect_failure(box.run(), "box")
    assert box.server1()[-1] == label and box.ssh_calls() == []


def test_one_second_before_the_box_the_next_call_still_starts_with_a_timeout_of_one_second(box):
    cached(box)
    box.answer(advance={"marker": BOX - 1}, rc={"has-image": 2})
    expect_failure(box.run(), "image-ask")
    assert box.kinds() == ["marker", "versions", "has-image"]
    assert [(w, s) for w, s, used in box.timeouts() if used == BOX - 1] == [("ssh", "1"), ("ssh", "1")]


def test_every_call_is_bounded_by_the_seconds_left_and_both_sides_of_a_stream_get_the_same(shared):
    box, done = shared("advances")
    expect_success(done)
    groups: dict[int, list[tuple[str, str]]] = {}
    for wrapped, seconds, used in box.timeouts():
        groups.setdefault(used, []).append((wrapped, seconds))
    assert sorted(groups) == [0, 300, 420, 480, 525, 725]
    for used, found in groups.items():
        assert {seconds for _, seconds in found} == {str(BOX - used)}, (used, found)
    assert box.timeouts()[0][:2] == ("git", "1500")
    counted = {used: dict(Counter(wrapped for wrapped, _ in found)) for used, found in groups.items()}
    assert counted == {
        0: {"git": 2, "docker": 1, "ssh": 1}, 300: {"ssh": 3, "git": 1}, 420: {"ssh": 2, "docker": 1},
        480: {"ssh": 4, "git": 1}, 525: {"ssh": 3}, 725: {"ssh": 12},
    }


def test_a_platform_build_that_never_gets_ready_is_polled_every_ten_seconds_until_the_box(box):
    no_platform(box)
    box.answer(ready_after=None, advance={"build-platform": 1397})
    expect_failure(box.run(), "box")
    kinds = box.kinds()
    assert box.sleeps() == ["10"] * 10 + ["3"] and box.elapsed() == BOX
    assert kinds.count("build-platform") == 1 and kinds[kinds.index("build-platform") + 1] == "has-image"
    assert kinds[-1] == "has-image" and not AFTER_BOX & set(kinds)
    assert box.remote("has-image")[2:] == [INSPECT_PLATFORM] * 11
    nothing_stopped(box)


STATUS_CALLS = [
    ("full", "git toplevel", None), ("full", "git head", None), ("full", "docker inspect", None),
    ("full", "ship-scripts", "ship-scripts"), ("full", "git archive scripts", "ship-scripts"),
    ("full", "publish-scripts", "publish-scripts"),
    ("full", "docker save", "load"), ("full", "load", "load"), ("full", "ship-platform", "ship-platform"),
    ("full", "git archive commit", "ship-platform"), ("full", "publish-platform", "publish-platform"),
    ("full", "build-platform", "build-platform"), ("full", "has-image#3", "has-image"),
    ("full", "rm-archive", "rm-archive"),
    ("cached", "marker", "marker"), ("cached", "versions#1", "versions"), ("cached", "has-image#1", "has-image"),
    ("cached", "has-image#2", "has-image"),
    ("cached", "tag", "tag"), ("cached", "build-proxy", "build-proxy"), ("cached", "converge", "converge"),
    ("cached", "copy-page-check", "copy-page-check"), ("cached", "publish-page-check", "publish-page-check"),
    ("cached", "rm-scripts-new", "rm-scripts-new"), ("cached", "rm-page-check", "rm-page-check"),
    ("cached", "rm-staging-tar", "rm-staging-tar"),
    ("cached", "versions#2", "versions"), ("cached", "list-tags", "list-tags"), ("cached", "remove-tag", "remove-tag"),
    ("current", "page-check-present", "page-check-present"), ("current", "rm-archive", "rm-archive"),
]
STATUS_CASES = [(124, *row) for row in STATUS_CALLS] + [(137, *row) for row in STATUS_CALLS[::4]]


@pytest.mark.parametrize("status, scenario, key, last", STATUS_CASES,
                         ids=[f"{row[0]}:{row[2]}" for row in STATUS_CASES])
def test_status_124_or_137_of_any_call_is_the_box_even_when_the_clock_did_not_move(box, status, scenario, key, last):
    SCENARIOS[scenario](box)
    box.answer(tags=RETENTION_TAGS, rc={key: status})
    expect_failure(box.run(), "box")
    assert box.elapsed() == 0 and (box.kinds()[-1] if box.kinds() else None) == last
    nothing_stopped(box)


LATE_FAILURES = [
    ("full", "ship-scripts", 1), ("full", "publish-scripts", 1), ("full", "publish-platform", 1),
    ("full", "build-platform", 1), ("cached", "versions#1", 1),
    ("cached", "has-image#1", 2), ("cached", "converge", 1), ("cached", "copy-page-check", 1),
    ("cached", "publish-page-check", 1), ("cached", "remove-tag", 1), ("full", "rm-archive", 1),
    ("cached", "rm-scripts-new", 1), ("cached", "rm-page-check", 1), ("cached", "rm-staging-tar", 1),
    ("current", "rm-archive", 1),
]


@pytest.mark.parametrize("scenario, key, rc", LATE_FAILURES, ids=[f"{s}:{k}" for s, k, _ in LATE_FAILURES])
def test_a_failure_with_the_box_over_is_the_box_and_not_the_reason_of_its_step(box, scenario, key, rc):
    SCENARIOS[scenario](box)
    box.answer(tags=RETENTION_TAGS, rc={key: rc}, advance={key: BOX})
    expect_failure(box.run(), "box")
    assert box.kinds()[-1] == key.split("#")[0]


def test_a_failure_one_second_before_the_box_keeps_its_own_reason(box):
    cached(box)
    box.answer(rc={"converge": 1}, advance={"converge": BOX - 1})
    expect_failure(box.run(), "converge")
    assert box.elapsed() == BOX - 1 and box.kinds()[-1] == "converge"


LATE = [
    ("full", "ship-scripts"), ("full", "publish-scripts"), ("full", "load"), ("full", "ship-platform"),
    ("full", "publish-platform"), ("full", "build-platform"),
    ("full", "has-image#3"), ("full", "rm-archive"), ("cached", "marker"), ("cached", "has-image#1"),
    ("cached", "has-image#2"), ("cached", "tag"), ("cached", "build-proxy"), ("cached", "converge"),
    ("cached", "copy-page-check"), ("cached", "publish-page-check"), ("cached", "rm-scripts-new"),
    ("cached", "rm-page-check"), ("cached", "rm-staging-tar"), ("cached", "versions#2"), ("cached", "list-tags#2"),
    ("cached", "remove-tag#1"), ("current", "page-check-present"), ("current", "rm-archive"),
]


@pytest.mark.parametrize("scenario, key", LATE, ids=[f"{s}:{k}" for s, k in LATE])
def test_a_call_that_succeeds_after_the_box_ran_out_is_the_box_and_nothing_further_starts(box, scenario, key):
    SCENARIOS[scenario](box)
    box.answer(tags=RETENTION_TAGS, advance={key: BOX + 1})
    expect_failure(box.run(), "box")
    assert box.kinds()[-1] == key.split("#")[0]
    nothing_stopped(box)


# ---- E. the stand-ins keep their state safe when several run at once --------------------------------


def test_stand_in_calls_at_the_same_time_never_show_half_a_state_and_lose_no_update(box):
    workers, rounds = 4, 12
    box.answer(advance={"has-image": 1})
    env = {"PATH": str(box.bin), "HOME": str(box.top), "STAND_IN_PAUSE": "0.003"}
    loop = 'i=0; while [ $i -lt %d ]; do ssh "$1" "$2" </dev/null >/dev/null 2>&1; i=$((i+1)); done' % rounds
    stop, broken, reads = threading.Event(), [], [0]

    def reader() -> None:
        while not stop.is_set():
            for name, parse in (("state.json", json.loads), ("clock", int)):
                try:
                    text = (box.top / name).read_text(encoding="utf-8")
                except FileNotFoundError:
                    continue
                reads[0] += 1
                try:
                    parse(text)
                except ValueError:
                    broken.append((name, text))

    watcher = threading.Thread(target=reader)
    watcher.start()
    try:
        running = [subprocess.Popen(["/bin/sh", "-c", loop, "sh", RUNNER, INSPECT_APP], env=env, stdin=subprocess.DEVNULL)
                   for _ in range(workers)]
        assert [process.wait(timeout=120) for process in running] == [0] * workers
    finally:
        stop.set()
        watcher.join()
    total = workers * rounds
    assert broken == [], f"a reader saw a half-written file: {broken[:1]}"
    assert reads[0] > 0, "the reader never read"
    counted = json.loads((box.top / "state.json").read_text(encoding="utf-8"))["count"]["has-image"]
    assert counted == total, f"updates were lost in state.json: {counted} of {total}"
    assert len(box.of("has-image")) == total, "calls are missing from the log"
    assert box.elapsed() == total, f"updates were lost in the clock: {box.elapsed()} of {total}"
