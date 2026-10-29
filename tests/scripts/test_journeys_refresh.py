"""Tests for the journeys refresh script (card 3: refresh_stack.sh on Server 1).

refresh_stack.sh runs for real, from a temporary copy of the journeys folder inside a temporary
"checkout", against stand-ins that sit alone on PATH: ssh, docker, git, timeout, date and sleep (plus
symlinks to bash and dirname). Every stand-in logs its call to one file, in order, and answers from
answers.json. Only the fake ssh (Server 2) keeps state: state.json is read and rewritten inside an
exclusive flock and is published by rename, so no process ever sees half of it; the git and docker of
Server 1 never touch it. The fake ssh records [destination, command, stdin size and sha256], answers by
the kind of command and may advance the clock; it runs three commands for real in a temporary Server 2
folder (the marker read, the scripts swap, and the page check copy against a fake docker that maps each
container to a folder). The clock is a number in a file that is published by rename: date +%s prints it,
sleep N adds N and returns at once, timeout logs its seconds and runs the wrapped command. No test reaches
a real host, docker or ssh. Runs of the script are shared between tests that look at the same scenario.
"""

from __future__ import annotations

import json
import os
import re
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
PROXY_TAG = "0123456789ab"
CA = "cacacacacaca"
APP_TAG = f"{REVISION}-{CA}"
PLATFORM_REF = f"raphael-journeys-platform:{COMMIT}"
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


def end(rc, out=None, data=None):
    # every stand-in is noisy with a value that must never come out of the script
    if data is not None:
        sys.stdout.buffer.write(data)
    elif out is not None:
        sys.stdout.write(out)
    else:
        sys.stdout.write("leak: %s\n" % SECRET)
    sys.stderr.write("leak: %s\n" % SECRET)
    sys.stdout.flush()
    sys.exit(rc)


def make_tar(files):
    import io, tarfile
    buf, seen = io.BytesIO(), set()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for path in sorted(files):
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
    return buf.getvalue()


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
    data = make_tar(A["scripts_files"] if label.endswith("scripts") else A["commit_files"])
    record(label, dir=where, sha=sha(data))
    end(0, data=data)

if name == "docker":
    if args[:1] == ["inspect"]:
        record("docker inspect")
        bump_clock(A["advance"].get("docker inspect", 0))
        end(A["rc"].get("docker inspect", A["inspect"]["rc"]), out=A["inspect"]["out"])
    if args[:1] == ["save"]:
        data = A["save_data"].encode("utf-8")
        rc = A["rc"].get("docker save", 0)
        record("docker save", sha=sha(data))
        end(rc, data=b"" if rc else data)
    record("docker other")
    end(98, out="")
'''

STAND_IN += r'''
if name == "ssh":
    import shlex
    command = args[-1]
    data = b"" if sys.stdin.isatty() else sys.stdin.buffer.read()
    D, B = A["s2_dir"], A["browser"]
    exact = {
        f"cat '{D}/.refresh-commit'": "marker",
        f"cd '{D}' && ./stack_up.sh versions": "versions",
        f"cd '{D}' && ./stack_up.sh build proxy": "build-proxy",
        f"docker exec '{B}' test -d '/opt/raphael-journeys/tools/page-check'": "page-check-present",
        "docker load": "load",
    }
    prefixes = [
        ("docker image inspect ", "has-image"), ("docker tag ", "tag"), ("cat > ", "ship-platform"),
        (f"cd '{D}' && ./stack_up.sh build platform ", "build-platform"),
        (f"cd '{D}' && ./stack_up.sh converge ", "converge"), ("rm ", "rm-archive"),
        ("docker image ls ", "list-tags"), ("docker image rm ", "remove-tag"),
    ]
    kind = exact.get(command)
    if kind is None:
        if " tar -xf - -C " in command:
            kind = "ship-scripts"
        elif "docker cp " in command:
            kind = "copy-page-check"
        else:
            kind = next((k for p, k in prefixes if command.startswith(p)), "other")
    try:
        words = shlex.split(command)
    except ValueError:
        words = []
    last = words[-1] if words else ""
    arg = {"has-image": last, "tag": last, "remove-tag": last,
           "list-tags": words[3] if len(words) > 3 else ""}.get(kind, "")
    label = (kind + " " + arg).strip()
    record(kind, remote=command, dest=args[0], nargs=len(args), stdin_size=len(data),
           stdin_sha=sha(data), secret_in_stdin=SECRET.encode("utf-8") in data, arg=arg)
    real, rc, out = False, 0, None
    with locked(state_path):
        if os.path.exists(state_path):
            with open(state_path, encoding="utf-8") as handle:
                state = json.load(handle)
        else:
            state = {"images": list(A["images"]), "tags": {k: list(v) for k, v in A["tags"].items()},
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
        elif forced is not None:
            rc = forced
        elif kind in ("marker", "ship-scripts", "copy-page-check"):
            real = True
        elif kind == "has-image":
            ref = "raphael-journeys-platform:" + A["head"]
            if (arg == ref and A["ready_after"] is not None and state["built_at"] is not None
                    and now() - state["built_at"] >= A["ready_after"] and ref not in state["images"]):
                state["images"].append(ref)
            rc = 0 if arg in state["images"] else 1
        elif kind == "load":
            state["images"].append(A["app_image_id"])
        elif kind == "tag":
            known = words[2] in state["images"]
            if known:
                state["images"].append(words[3])
            rc = 0 if known else 1
        elif kind == "remove-tag":
            repo, tag = arg.split(":", 1)
            if tag in state["tags"].get(repo, []):
                state["tags"][repo].remove(tag)
        elif kind not in ("page-check-present", "ship-platform", "build-platform", "rm-archive"):
            rc = 98
        if kind == "build-platform" and rc == 0:
            state["built_at"] = now() + advance
        if kind == "converge" and rc == 0:
            state["converged"] = True
        pause()
        publish(state_path, json.dumps(state))
    if real:
        import subprocess
        env = {"PATH": os.path.join(top, "s2bin"), "HOME": top}
        done = subprocess.run(["/bin/sh", "-c", command], env=env, input=data, capture_output=True, cwd=top)
        rc, out = done.returncode, (done.stdout.decode("utf-8") if kind == "marker" else None)
    bump_clock(advance)
    end(rc, out=out)

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
        sys.exit(failing)
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


FULL_RUN = [
    "marker", "ship-scripts", "has-image", "load", "tag", "has-image", "ship-platform", "build-platform",
    "has-image", "rm-archive", "build-proxy", "converge", "copy-page-check", "versions", *["list-tags"] * 5,
]
CURRENT_RUN = ["marker", "versions", "page-check-present", "versions", *["list-tags"] * 5]
CACHED_RUN = [
    "marker", "versions", "has-image", "tag", "has-image", "build-proxy", "converge", "copy-page-check",
    "versions", *["list-tags"] * 5,
]
SHIPPING = {"ship-scripts", "load", "ship-platform", "build-platform"}


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
        for name in ("sh", "tar", "mkdir", "rm", "cat"):
            (self.s2bin / name).symlink_to(shutil.which(name))
        (self.s2bin / "mv").write_text(
            f'#!/bin/sh\ncase "$1" in *page-check.new) if [ -e "{top / "mv-fails"}" ]; then exit 1; fi ;; esac\n'
            f'exec "{shutil.which("mv")}" "$@"\n', encoding="utf-8")
        (self.s2bin / "mv").chmod(0o755)
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
            "images": [], "tags": {}, "ready_after": 0, "rc": {}, "advance": {},
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
            merged = key in ("rc", "out", "advance", "inspect")
            self.answers[key] = {**self.answers[key], **value} if merged else value
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
        done = subprocess.run(
            [str(copy), *argv], env=env, input="STDIN-SENTINEL\n", capture_output=True,
            encoding="utf-8", text=True, timeout=180, check=False,
        )
        self.audit()
        return done

    def audit(self) -> None:
        """Holds for every call of every test: no value, no forbidden word, JOURNEY_RUNNER_SSH alone, and
        every git, docker and ssh call sits behind a timeout that was given exactly the seconds left."""
        calls = self.calls()
        for call in calls:
            assert SECRET not in json.dumps(call), f"a value reached a call: {call['call']}"
            assert not call.get("secret_in_stdin"), "a value was streamed"
            words = forbidden(" ".join(call["argv"][1:]))
            assert not words, f"{words} in {call['argv']}"
            if call["argv"][0] == "ssh":
                assert call["nargs"] == 2 and call["dest"] == self.destination, call["argv"]
            if call["call"] == "timeout":
                seconds = call["argv"][1]
                assert re.fullmatch(r"[1-9][0-9]*", seconds), call["argv"]
                assert int(seconds) == BOX - (call["now"] - CLOCK_START), (call["argv"], call["now"] - CLOCK_START)
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

    def stdin_calls(self) -> list[str]:
        return [call["call"] for call in self.ssh_calls() if call["stdin_size"]]

    def cmd(self, tail: str) -> str:
        return f"cd '{self.s2dir}' && {tail}"

    def archive(self, commit: str = COMMIT) -> str:
        return f"{self.s2dir}.platform-{commit}.tar"

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
    assert box.remote("marker") == [f"cat '{box.s2dir}/.refresh-commit'"]


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
    assert box.remote("marker") == [f"cat '{box.s2dir}/.refresh-commit'"]
    assert box.remote("versions") == [box.cmd("./stack_up.sh versions")] * 2
    assert box.remote("page-check-present") == [f"docker exec '{box.browser_c}' test -d '{BROWSER_PAGE_CHECK}'"]
    assert box.server1() == ["git toplevel", "git head", "docker inspect"]
    assert all(call["stdin_size"] == 0 for call in box.ssh_calls())
    assert box.sleeps() == [] and box.of("rm-archive") == []
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


MARKERS = {
    "missing": None, "other commit": OTHER_COMMIT + "\n", "empty": "", "two lines": COMMIT + "\n" + COMMIT + "\n",
    "text after the commit": COMMIT + " extra\n", "short": COMMIT[:39] + "\n", "a folder": "folder",
    "ssh fails": "ssh",
}


@pytest.mark.parametrize("text", list(MARKERS.values()), ids=list(MARKERS))
def test_a_marker_that_is_missing_unreadable_or_not_the_commit_is_not_current_and_ships_the_scripts_once(box, text):
    if text == "folder":
        box.set_marker(None, folder=True)
    elif text == "ssh":
        box.answer(rc={"marker": 255})
    else:
        box.set_marker(text)
    box.answer(rc={"ship-scripts": 1})
    expect_failure(box.run(), "ship-scripts")
    assert box.kinds() == ["marker", "ship-scripts"]
    assert box.stdin_calls() == ["ship-scripts"] and box.server1()[-1] == "git archive scripts"
    assert len(box.of("git archive scripts")) == 1


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
    assert kinds[:2] == ["marker", "ship-scripts"]
    assert kinds.index("ship-scripts") < min(kinds.index(k) for k in ("load", "ship-platform", "build-platform"))
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


def test_the_ship_command_writes_the_marker_into_the_new_folder_after_the_extract_and_before_the_renames(shared):
    box, _ = shared("full")
    (command,) = box.remote("ship-scripts")
    write = f"echo '{COMMIT}' > '{box.s2dir}.new/.refresh-commit'"
    assert command.count(write) == 1
    assert command.index("tar -xf -") < command.index(write) < command.index("mv ")


def test_an_existing_folder_is_kept_as_the_previous_copy_and_the_new_one_holds_the_marker(box):
    parent = box.s2dir.parent
    box.set_marker(OTHER_COMMIT + "\n")
    (box.s2dir / "old.txt").write_text("OLD\n", encoding="utf-8")
    box.answer(rc={"has-image": 2})
    expect_failure(box.run(), "image-ask")
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
    box.answer(rc={"has-image": 2})
    expect_failure(box.run(), "image-ask")
    assert tree(parent / "raphael-journeys.previous") == {"gen1.txt": "ONE\n", ".refresh-commit": OTHER_COMMIT + "\n"}
    assert tree(box.s2dir) == {**expected_scripts(box), ".refresh-commit": COMMIT + "\n"}
    assert not (parent / "raphael-journeys.new").exists()
    (box.s2dir / ".refresh-commit").write_text(OTHER_COMMIT + "\n", encoding="utf-8")
    shipped = tree(box.s2dir)
    box.answer(scripts_files={"scripts/journeys/stack_up.sh": "# third generation\n"})
    expect_failure(box.run(), "image-ask")
    assert tree(parent / "raphael-journeys.previous") == shipped
    assert tree(box.s2dir) == {"stack_up.sh": "# third generation\n", ".refresh-commit": COMMIT + "\n"}


@pytest.mark.parametrize("rc", [{"ship-scripts": 1}, {"git archive scripts": 1}], ids=["ssh", "git"])
def test_scripts_that_cannot_be_shipped_end_the_run(box, rc):
    box.answer(rc=rc)
    expect_failure(box.run(), "ship-scripts")
    assert box.kinds() == ["marker", "ship-scripts"] and not box.s2dir.exists()


def test_a_real_failure_of_the_scripts_swap_ends_the_run_and_changes_nothing(box):
    broken = box.top / "no-such-parent" / "raphael-journeys"
    box.answer(s2_dir=str(broken))
    expect_failure(box.run(JOURNEY_S2_SCRIPTS_DIR=str(broken)), "ship-scripts")
    assert not broken.parent.exists() and box.kinds() == ["marker", "ship-scripts"]


def test_every_remote_command_parses_as_sh(shared):
    box, done = shared("full")
    expect_success(done)
    assert box.kinds() == FULL_RUN
    for call in box.ssh_calls():
        parsed = subprocess.run(["sh", "-n", "-c", call["remote"]], capture_output=True, encoding="utf-8", text=True)
        assert parsed.returncode == 0, (call["remote"], parsed.stderr)


# ---- b. the owner app image: loaded only when Server 2 lacks the id --------------------------------


def test_a_known_app_image_is_not_loaded_but_still_tagged(shared):
    box, done = shared("no-platform")
    expect_success(done)
    assert "load" not in box.kinds() and "docker save" not in box.server1()
    assert box.remote("has-image")[0] == f"docker image inspect '{IMAGE_ID}'"
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


@pytest.mark.parametrize("rc", [2, 125, 255])
def test_any_exit_but_0_and_1_of_the_image_question_is_a_failure(box, rc):
    box.answer(rc={f"has-image {IMAGE_ID}": rc})
    expect_failure(box.run(), "image-ask")
    assert box.kinds() == ["marker", "ship-scripts", "has-image"] and box.of("docker save") == []


@pytest.mark.parametrize("rc", [{"load": 1}, {"tag": 1}, {"docker save": 1}], ids=["load", "tag", "save"])
def test_a_failed_load_tag_or_save_ends_with_its_reason(box, rc):
    box.answer(rc=rc)
    expect_failure(box.run(), "app-image")
    assert not {"build-platform", "converge"} & set(box.kinds())


# ---- c. the platform image ------------------------------------------------------------------------


def test_both_images_known_and_the_marker_equal_cross_nothing_and_the_stack_still_converges(shared):
    box, done = shared("cached")
    expect_success(done)
    assert box.kinds() == CACHED_RUN
    assert box.server1() == ["git toplevel", "git head", "docker inspect"]
    assert box.stdin_calls() == [] and not (SHIPPING | {"rm-archive"}) & set(box.kinds())
    assert box.remote("has-image") == [f"docker image inspect '{IMAGE_ID}'", f"docker image inspect '{PLATFORM_REF}'"]
    assert box.remote("tag") == [f"docker tag '{IMAGE_ID}' 'raphael-journeys-app-base:{REVISION}'"]
    assert box.sleeps() == [] and not box.staging.parent.joinpath("raphael-journeys.new").exists()


def test_a_missing_platform_image_is_shipped_as_one_file_under_a_temporary_name_and_built(shared):
    box, done = shared("no-platform")
    expect_success(done)
    part, final = f"{box.archive()}.part", box.archive()
    assert box.remote("ship-platform") == [f"cat > '{part}' && mv '{part}' '{final}'"]
    assert box.remote("build-platform") == [box.cmd(f"./stack_up.sh build platform '{COMMIT}' '{final}'")]
    (ship,) = box.of("ship-platform")
    (git,) = box.of("git archive commit")
    assert ship["stdin_sha"] == git["sha"] and git["argv"][1:] == ["-C", str(box.checkout), "archive", "--format=tar", COMMIT]
    kinds = box.kinds()
    assert kinds.index("ship-platform") + 1 == kinds.index("build-platform") and kinds.count("build-platform") == 1
    assert box.server1() == ["git toplevel", "git head", "docker inspect", "git archive commit"]


def test_the_wait_checks_at_once_then_every_ten_seconds_and_the_archive_is_removed_when_the_image_is_there(box):
    no_platform(box)
    box.answer(ready_after=60, rc={"build-proxy": 1})
    expect_failure(box.run(), "proxy-tag")
    kinds = box.kinds()
    assert kinds[kinds.index("build-platform") + 1] == "has-image"
    # the question before the shipping, then the first look at once and one more look after each of the 6 sleeps
    assert box.remote("has-image")[1:] == [f"docker image inspect '{PLATFORM_REF}'"] * 8
    assert box.sleeps() == ["10"] * 6 and box.elapsed() == 60
    assert kinds.index("build-platform") < kinds.index("rm-archive") < kinds.index("build-proxy")
    assert kinds[kinds.index("rm-archive") - 1] == "has-image"
    assert box.remote("rm-archive") == [f"rm '{box.archive()}'"]


def test_a_platform_question_that_fails_ends_the_run_before_shipping(box):
    no_platform(box)
    box.answer(rc={f"has-image {PLATFORM_REF}": 2})
    expect_failure(box.run(), "image-ask")
    assert "ship-platform" not in box.kinds() and box.of("git archive commit") == []


@pytest.mark.parametrize("rc", [{"ship-platform": 1}, {"build-platform": 1}, {"git archive commit": 1}],
                         ids=["ship", "build", "git"])
def test_a_platform_archive_or_build_that_fails_ends_the_run(box, rc):
    no_platform(box)
    box.answer(rc=rc)
    expect_failure(box.run(), "platform-build")
    assert not {"build-proxy", "rm-archive", "converge"} & set(box.kinds())


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


def test_the_page_check_is_copied_by_one_command_joined_with_and_after_converge_and_before_the_retention(shared):
    box, done = shared("cached")
    expect_success(done)
    kinds = box.kinds()
    assert kinds.index("converge") + 1 == kinds.index("copy-page-check") == kinds.index("versions", kinds.index("converge")) - 1
    (command,) = box.remote("copy-page-check")
    tar, dest = f"{box.s2dir}.page-check.tar", BROWSER_PAGE_CHECK
    produce = f"docker cp '{box.app_c}:{OPT}/.' - > '{tar}'"
    consume = f"docker cp - '{box.browser_c}:{dest}.new' < '{tar}'"
    clean = f"if [ -e '{tar}' ]; then rm '{tar}'; fi"
    assert command.count("docker cp ") == 2 and command.count(" && ") == 4 and box.of("copy-page-check")[0]["stdin_size"] == 0
    assert not re.search(r"(?<!\|)\|(?!\|)", command), "the copy must not be a pipeline"
    assert command.index(produce) < command.index(consume) < command.rindex("mv ") < command.index(clean)
    assert command.endswith(clean) and command.startswith(f"docker exec '{box.browser_c}' sh -c ")


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
    (command,) = box.remote("copy-page-check")
    assert "'owner-app-renamed:" in command and "'browser-renamed:" in command and "raphael-journeys-" not in command
    assert tree(box.old_copy) == source and "browser-renamed" in str(box.old_copy)


def _no_source(box):
    shutil.rmtree(box.source_folder)


def _empty_source(box):
    for path in list(box.source_folder.rglob("*"))[::-1]:
        path.unlink() if path.is_file() else path.rmdir()


def _no_browser(box):
    shutil.rmtree(box.fs / box.browser_c)


PAGE_FAILS = {
    "no source": _no_source, "empty source": _empty_source, "no browser": _no_browser,
    "producer fails with 23 after a valid partial archive": lambda box: box.answer(rc={"s2docker cp-out": 23}),
    "consumer fails": lambda box: box.answer(rc={"s2docker cp-in": 1}),
    "swap fails and rolls back": lambda box: (box.top / "mv-fails").write_text("x", encoding="utf-8"),
    "forced failure": lambda box: box.answer(rc={"copy-page-check": 1}),
}


@pytest.mark.parametrize("setup", list(PAGE_FAILS.values()), ids=list(PAGE_FAILS))
def test_a_failed_copy_ends_the_run_keeps_the_old_copy_untouched_and_removes_nothing(box, setup):
    cached(box)
    box.answer(tags=RETENTION_TAGS)
    setup(box)
    expect_failure(box.run(), "page-check")
    if (box.fs / box.browser_c).exists():
        assert tree(box.old_copy) == {"old.txt": "OLD\n"}
    assert box.kinds()[-1] == "copy-page-check" and box.of("remove-tag") == [] and box.of("list-tags") == []


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
    assert box.kinds()[:4] == CURRENT_RUN[:4]
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
    assert box.kinds() == CURRENT_RUN[:4] and stored_tags(box) == RETENTION_TAGS


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


# ---- A. one deadline of 25 minutes for the whole run -----------------------------------------------

AFTER_BOX = {"build-proxy", "converge", "copy-page-check", "rm-archive", "list-tags", "remove-tag"}


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
    assert not {"copy-page-check", "list-tags", "remove-tag"} & set(kinds)
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
    ("has-image#1", "docker save", {"load", "tag", "ship-platform", "build-platform"}),
    ("has-image#2", "git archive commit", {"ship-platform", "build-platform"}),
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
        0: {"git": 2, "docker": 1, "ssh": 1}, 300: {"git": 1, "ssh": 1}, 420: {"ssh": 2, "docker": 1},
        480: {"ssh": 4, "git": 1}, 525: {"ssh": 4}, 725: {"ssh": 7},
    }


def test_a_platform_build_that_never_gets_ready_is_polled_every_ten_seconds_until_the_box(box):
    no_platform(box)
    box.answer(ready_after=None, advance={"build-platform": 1397})
    expect_failure(box.run(), "box")
    kinds = box.kinds()
    assert box.sleeps() == ["10"] * 10 + ["3"] and box.elapsed() == BOX
    assert kinds.count("build-platform") == 1 and kinds[kinds.index("build-platform") + 1] == "has-image"
    assert kinds[-1] == "has-image" and not AFTER_BOX & set(kinds)
    assert box.remote("has-image")[2:] == [f"docker image inspect '{PLATFORM_REF}'"] * 11
    nothing_stopped(box)


STATUS_CALLS = [
    ("full", "git toplevel", None), ("full", "git head", None), ("full", "docker inspect", None),
    ("full", "ship-scripts", "ship-scripts"), ("full", "git archive scripts", "ship-scripts"),
    ("full", "docker save", "load"), ("full", "load", "load"), ("full", "ship-platform", "ship-platform"),
    ("full", "git archive commit", "ship-platform"), ("full", "build-platform", "build-platform"),
    ("full", "has-image#3", "has-image"), ("full", "rm-archive", "rm-archive"),
    ("cached", "marker", "marker"), ("cached", "versions#1", "versions"), ("cached", "has-image#1", "has-image"),
    ("cached", "tag", "tag"), ("cached", "build-proxy", "build-proxy"), ("cached", "converge", "converge"),
    ("cached", "copy-page-check", "copy-page-check"), ("cached", "versions#2", "versions"),
    ("cached", "list-tags", "list-tags"), ("cached", "remove-tag", "remove-tag"),
    ("current", "page-check-present", "page-check-present"),
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
    ("full", "ship-scripts", 1), ("full", "build-platform", 1), ("cached", "versions#1", 1),
    ("cached", "has-image#1", 2), ("cached", "converge", 1), ("cached", "copy-page-check", 1),
    ("cached", "remove-tag", 1),
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
    ("full", "ship-scripts"), ("full", "load"), ("full", "ship-platform"), ("full", "build-platform"),
    ("full", "has-image#3"), ("full", "rm-archive"), ("cached", "marker"), ("cached", "has-image#1"),
    ("cached", "tag"), ("cached", "build-proxy"), ("cached", "converge"), ("cached", "copy-page-check"),
    ("cached", "versions#2"), ("cached", "list-tags#2"), ("cached", "remove-tag#1"),
    ("current", "page-check-present"),
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
    loop = 'i=0; while [ $i -lt %d ]; do ssh "$1" "docker image inspect img$i" </dev/null >/dev/null 2>&1; i=$((i+1)); done' % rounds
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
        running = [subprocess.Popen(["/bin/sh", "-c", loop, "sh", RUNNER], env=env, stdin=subprocess.DEVNULL)
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
