"""Tests for the journeys refresh script (card 3: refresh_stack.sh on Server 1).

refresh_stack.sh runs for real, from a temporary copy of the journeys folder inside a temporary
"checkout", against stand-ins that sit alone on PATH: ssh, docker, git, date and sleep (plus symlinks
to bash and dirname). Every stand-in logs its call to one file, in order, and answers from
answers.json. The docker of Server 1 (inspect, save) is kept apart from what the fake ssh does on
"Server 2": the fake ssh records [destination, command, stdin size and sha256], answers by the kind of
command and may advance the clock; it runs two commands for real in a temporary Server 2 folder (the
scripts swap, and the page check copy against a fake docker that maps each container to a folder). The
clock is a number in a file: date +%s prints it, sleep N adds N and returns at once. No test reaches a
real host, docker or ssh.
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
    "box": "the platform build was still running at the 25-minute box and keeps running",
    "proxy-tag": "the proxy tag was not reported",
    "converge": "the stack could not be converged",
    "page-check": "the page check could not be copied into the browser container",
    "prune": "old images could not be removed",
}
REPOSITORIES = [
    "raphael-journeys-proxy", "raphael-journeys-app-base", "raphael-journeys-app",
    "raphael-journeys-platform", "raphael-journeys-browser",
]

STAND_IN = r'''
import hashlib, io, json, os, sys
from pathlib import Path

top = Path(__file__).parent
name = Path(sys.argv[1]).name
args = sys.argv[2:]
A = json.loads((top / "answers.json").read_text(encoding="utf-8"))
SECRET = A["secret"]
log, clock, state_path = top / "calls.jsonl", top / "clock", top / "state.json"
state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {
    "images": list(A["images"]), "tags": {k: list(v) for k, v in A["tags"].items()}, "built_at": None,
}


def now():
    return int(clock.read_text(encoding="utf-8"))


def record(label, **extra):
    with open(log, "a", encoding="utf-8") as handle:
        handle.write(json.dumps({"call": label, "argv": [name, *args], **extra}) + "\n")


def end(rc, out=None, data=None):
    # every stand-in is noisy with a value that must never come out of the script
    state_path.write_text(json.dumps(state), encoding="utf-8")
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
    import tarfile
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
        end(rc, out="" if rc else out)
    data = make_tar(A["scripts_files"] if label.endswith("scripts") else A["commit_files"])
    record(label, dir=where, sha=hashlib.sha256(data).hexdigest())
    end(0, data=data)

if name == "docker":
    if args[:1] == ["inspect"]:
        record("docker inspect")
        end(A["inspect"]["rc"], out=A["inspect"]["out"])
    if args[:1] == ["save"]:
        data = A["save_data"].encode("utf-8")
        rc = A["rc"].get("docker save", 0)
        record("docker save", sha=hashlib.sha256(data).hexdigest())
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
           stdin_sha=hashlib.sha256(data).hexdigest(), secret_in_stdin=SECRET.encode("utf-8") in data, arg=arg)
    forced = A["rc"].get(label, A["rc"].get(kind))
    advance = A["advance"].get(kind, 0)
    parsed = ("versions", "build-proxy", "converge", "list-tags")

    def done(rc, out=None):
        clock.write_text(str(now() + advance), encoding="utf-8")
        if kind == "build-platform" and rc == 0:
            state["built_at"] = now()
        end(rc, out=out)

    if kind in parsed:
        text = ("".join(t + "\n" for t in state["tags"].get(arg, [])) if kind == "list-tags" else A["out"][kind])
        done(forced or 0, out=text)
    if forced is not None:
        done(forced)
    if kind in ("ship-scripts", "copy-page-check"):
        import subprocess
        env = {"PATH": str(top / "s2bin"), "HOME": str(top)}
        real = subprocess.run(["/bin/sh", "-c", command], env=env, input=data, capture_output=True, cwd=str(top))
        done(real.returncode)
    if kind == "has-image":
        ref = "raphael-journeys-platform:" + A["head"]
        if (arg == ref and A["ready_after"] is not None and state["built_at"] is not None
                and now() - state["built_at"] >= A["ready_after"] and ref not in state["images"]):
            state["images"].append(ref)
        done(0 if arg in state["images"] else 1)
    if kind == "load":
        state["images"].append(A["app_image_id"])
        done(0)
    if kind == "tag":
        known = words[2] in state["images"]
        if known:
            state["images"].append(words[3])
        done(0 if known else 1)
    if kind == "remove-tag":
        repo, tag = arg.split(":", 1)
        if tag in state["tags"].get(repo, []):
            state["tags"][repo].remove(tag)
        done(0)
    done(0 if kind in ("page-check-present", "ship-platform", "build-platform", "rm-archive") else 98)

if name == "s2-docker":
    # Server 2 docker as far as the page check copy needs it: each container is a folder of s2fs
    fs = top / "s2fs"

    def host(container, path):
        return fs / container / path.lstrip("/")

    record("s2docker")
    if args[0] == "exec":
        import subprocess
        if not (fs / args[1]).is_dir():
            sys.exit(1)
        mapped = [str(host(args[1], a)) if a.startswith("/") else a for a in args[2:]]
        sys.exit(subprocess.run(mapped).returncode)
    if args[0] == "cp" and args[2] == "-":
        import tarfile
        container, src = args[1].split(":", 1)
        base = host(container, src)
        if not base.is_dir():
            sys.exit(1)
        with tarfile.open(fileobj=sys.stdout.buffer, mode="w|") as tar:
            for item in sorted(base.rglob("*")):
                tar.add(item, arcname="./" + str(item.relative_to(base)), recursive=False)
        sys.exit(0)
    if args[0] == "cp" and args[1] == "-":
        import tarfile
        container, dst = args[2].split(":", 1)
        raw = sys.stdin.buffer.read()
        if A["rc"].get("s2docker cp-in") or not host(container, dst).is_dir():
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
    "versions", "ship-scripts", "has-image", "load", "tag", "has-image", "ship-platform", "build-platform",
    "has-image", "rm-archive", "build-proxy", "converge", "copy-page-check", *["list-tags"] * 5,
]


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
        for name in ("sh", "tar", "mkdir", "rm", "mv", "cat"):
            (self.s2bin / name).symlink_to(shutil.which(name))
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
            f'#!/bin/sh\nread now < "{clock}"\necho $((now + $1)) > "{clock}"\n'
            f'printf \'{{"call": "sleep", "argv": ["sleep", "%s"]}}\\n\' "$1" >> "{log}"\n'
            f'echo "leak: {SECRET}"\n{leak}\n',
            encoding="utf-8",
        )
        for tool in ("date", "sleep"):
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
        """Holds for every call of every test: no value, no forbidden word, JOURNEY_RUNNER_SSH alone."""
        for call in self.calls():
            assert SECRET not in json.dumps(call), f"a value reached a call: {call['call']}"
            assert not call.get("secret_in_stdin"), "a value was streamed"
            words = forbidden(" ".join(call["argv"][1:]))
            assert not words, f"{words} in {call['argv']}"
            if call["argv"][0] == "ssh":
                assert call["nargs"] == 2 and call["dest"] == self.destination, call["argv"]

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

    def elapsed(self) -> int:
        return int((self.top / "clock").read_text(encoding="utf-8")) - CLOCK_START

    def cmd(self, tail: str) -> str:
        return f"cd '{self.s2dir}' && {tail}"

    def archive(self, commit: str = COMMIT) -> str:
        return f"{self.s2dir}.platform-{commit}.tar"

    def converge_cmd(self, proxy_tag: str = PROXY_TAG) -> str:
        return self.cmd(f"./stack_up.sh converge '{proxy_tag}' '{REVISION}' '{COMMIT}' '{REVISION}'")


@pytest.fixture
def box(tmp_path: Path) -> Sandbox:
    if not all(shutil.which(tool) for tool in ("bash", "dirname", "sh", "tar", "mkdir", "rm", "mv", "cat")):
        pytest.skip("needs bash, tar and coreutils")
    return Sandbox(tmp_path)


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
def test_invalid_settings_refuse_before_any_call(box, changes):
    expect_failure(box.run(**changes), "config")
    assert box.calls() == []


@pytest.mark.parametrize("edit", [
    "missing", "not plain", "outside a section", "four containers", "no production", "two production",
    "odd production name",
])
def test_template_problems_refuse_before_any_call(box, edit):
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


def test_a_trailing_slash_of_the_folder_is_stripped(box):
    box.run(JOURNEY_S2_SCRIPTS_DIR=f"{box.s2dir}/")
    assert box.remote("versions") == [box.cmd("./stack_up.sh versions")]


@pytest.mark.parametrize("answers", [
    {"rc": {"git toplevel": 1}}, {"rc": {"git head": 1}}, {"head": "main"}, {"head": COMMIT.upper()},
    {"head": COMMIT[:39]}, {"head": COMMIT + "0"},
])
def test_unreadable_commit_refuses_before_any_ssh_call(box, answers):
    box.answer(**answers)
    expect_failure(box.run(), "commit")
    assert box.ssh_calls() == []


def test_git_is_asked_from_the_script_folder_and_archives_run_from_the_checkout_root(box):
    expect_success(box.run())
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
    box.run()
    assert box.kinds()[:1] == ["versions"]


def test_default_hermes_home_is_dot_hermes_under_home(box):
    (box.top / ".hermes").mkdir()
    box.write_state(COMMIT, box.top / ".hermes")
    (box.home / "gateway_state.json").unlink()
    box.run(HERMES_HOME=None)
    assert box.kinds()[:1] == ["versions"]


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


def test_server1_runs_only_docker_inspect_and_docker_save_of_the_inspected_id(box):
    expect_success(box.run())
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
    box.run()
    assert box.of("docker inspect")[0]["argv"][-1] == "renamed-production"


# ---- first look: the stack may already be current --------------------------------------------------


def test_a_current_stack_is_left_alone(box):
    box.answer(out={"versions": versions_text()})
    expect_success(box.run())
    assert box.kinds() == ["versions", "page-check-present"]
    assert box.remote("versions") == [box.cmd("./stack_up.sh versions")]
    assert box.remote("page-check-present") == [
        f"docker exec '{box.browser_c}' test -d '{BROWSER_PAGE_CHECK}'",
    ]
    assert box.server1() == ["git toplevel", "git head", "docker inspect"]
    assert all(call["stdin_size"] == 0 for call in box.ssh_calls())
    assert not box.s2dir.exists() and box.sleeps() == []


def test_matching_versions_without_the_page_check_are_not_current(box):
    box.answer(out={"versions": versions_text()}, rc={"page-check-present": 1})
    expect_success(box.run())
    assert box.kinds()[:3] == ["versions", "page-check-present", "ship-scripts"] and "converge" in box.kinds()


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
def test_versions_that_differ_or_cannot_be_read_start_the_full_run(box, text, rc):
    box.answer(out={"versions": text}, rc={"versions": rc})
    expect_success(box.run())
    assert box.kinds()[:2] == ["versions", "ship-scripts"] and "converge" in box.kinds()


# ---- a. the scripts folder is shipped and swapped in -----------------------------------------------


def expected_scripts(box: Sandbox) -> dict[str, str]:
    return {p.removeprefix("scripts/journeys/"): t for p, t in box.answers["scripts_files"].items()}


def test_scripts_are_shipped_first_among_the_shipping_steps(box):
    expect_success(box.run())
    kinds = box.kinds()
    assert kinds.index("ship-scripts") < min(kinds.index(k) for k in ("load", "ship-platform", "build-platform"))
    (ship,) = box.of("ship-scripts")
    (archive,) = box.of("git archive scripts")
    assert ship["stdin_sha"] == archive["sha"] and ship["stdin_size"] > 0
    assert archive["argv"][1:] == ["-C", str(box.checkout), "archive", "--format=tar", COMMIT, "scripts/journeys"]


def test_a_first_run_without_a_folder_creates_it_with_the_files_directly_inside(box):
    parent = box.s2dir.parent
    expect_success(box.run())
    assert tree(box.s2dir) == expected_scripts(box)
    assert os.access(box.s2dir / "stack_up.sh", os.X_OK)
    assert not (parent / "raphael-journeys.new").exists() and not (parent / "raphael-journeys.previous").exists()


def test_an_existing_folder_is_kept_as_the_previous_copy(box):
    parent = box.s2dir.parent
    box.s2dir.mkdir()
    (box.s2dir / "old.txt").write_text("OLD\n", encoding="utf-8")
    expect_success(box.run())
    assert tree(parent / "raphael-journeys.previous") == {"old.txt": "OLD\n"}
    assert tree(box.s2dir) == expected_scripts(box)
    assert not (parent / "raphael-journeys.new").exists()


def test_a_second_refresh_replaces_the_previous_copy_and_a_stale_new_folder(box):
    parent = box.s2dir.parent
    box.s2dir.mkdir()
    (box.s2dir / "gen1.txt").write_text("ONE\n", encoding="utf-8")
    for suffix, name, text in (("previous", "ancient.txt", "ANCIENT\n"), ("new", "junk.txt", "JUNK\n")):
        (parent / f"raphael-journeys.{suffix}").mkdir()
        (parent / f"raphael-journeys.{suffix}" / name).write_text(text, encoding="utf-8")
    expect_success(box.run())
    assert tree(parent / "raphael-journeys.previous") == {"gen1.txt": "ONE\n"}
    assert tree(box.s2dir) == expected_scripts(box) and not (parent / "raphael-journeys.new").exists()
    shipped = expected_scripts(box)
    box.answer(scripts_files={"scripts/journeys/stack_up.sh": "# third generation\n"})
    expect_success(box.run())
    assert tree(parent / "raphael-journeys.previous") == shipped
    assert tree(box.s2dir) == {"stack_up.sh": "# third generation\n"}


@pytest.mark.parametrize("rc", [{"ship-scripts": 1}, {"git archive scripts": 1}], ids=["ssh", "git"])
def test_scripts_that_cannot_be_shipped_end_the_run(box, rc):
    box.answer(rc=rc)
    expect_failure(box.run(), "ship-scripts")
    assert box.kinds() == ["versions", "ship-scripts"]


def test_a_real_failure_of_the_scripts_swap_ends_the_run_and_changes_nothing(box):
    broken = box.top / "no-such-parent" / "raphael-journeys"
    box.answer(s2_dir=str(broken))
    expect_failure(box.run(JOURNEY_S2_SCRIPTS_DIR=str(broken)), "ship-scripts")
    assert not broken.parent.exists() and box.kinds()[-1] == "ship-scripts"


def test_every_remote_command_parses_as_sh(box):
    expect_success(box.run())
    assert box.kinds() == FULL_RUN
    for call in box.ssh_calls():
        done = subprocess.run(["sh", "-n", "-c", call["remote"]], capture_output=True, encoding="utf-8", text=True)
        assert done.returncode == 0, (call["remote"], done.stderr)


# ---- b. the owner app image: loaded only when Server 2 lacks the id --------------------------------


def test_a_known_app_image_is_not_loaded_but_still_tagged(box):
    box.answer(images=[IMAGE_ID])
    expect_success(box.run())
    assert "load" not in box.kinds() and "docker save" not in box.server1()
    assert box.remote("has-image")[0] == f"docker image inspect '{IMAGE_ID}'"
    assert box.remote("tag") == [f"docker tag '{IMAGE_ID}' 'raphael-journeys-app-base:{REVISION}'"]
    assert [c["call"] for c in box.ssh_calls() if c["stdin_size"]] == ["ship-scripts", "ship-platform"]


def test_a_missing_app_image_is_saved_once_and_loaded_once_and_then_tagged(box):
    expect_success(box.run())
    (save,) = box.of("docker save")
    (load,) = box.of("load")
    assert save["argv"] == ["docker", "save", IMAGE_ID]
    assert load["remote"] == "docker load" and load["stdin_sha"] == save["sha"]
    assert load["stdin_size"] == len(box.answers["save_data"])
    kinds = box.kinds()
    assert kinds.index("has-image") < kinds.index("load") < kinds.index("tag")
    assert box.remote("tag") == [f"docker tag '{IMAGE_ID}' 'raphael-journeys-app-base:{REVISION}'"]
    assert [c["call"] for c in box.ssh_calls() if c["stdin_size"]] == ["ship-scripts", "load", "ship-platform"]


@pytest.mark.parametrize("rc", [2, 125, 255])
def test_any_exit_but_0_and_1_of_the_image_question_is_a_failure(box, rc):
    box.answer(rc={f"has-image {IMAGE_ID}": rc})
    expect_failure(box.run(), "image-ask")
    assert box.kinds() == ["versions", "ship-scripts", "has-image"] and box.of("docker save") == []


@pytest.mark.parametrize("rc", [{"load": 1}, {"tag": 1}, {"docker save": 1}], ids=["load", "tag", "save"])
def test_a_failed_load_tag_or_save_ends_with_its_reason(box, rc):
    box.answer(rc=rc)
    expect_failure(box.run(), "app-image")
    assert not {"build-platform", "converge"} & set(box.kinds())


# ---- c. the platform image ------------------------------------------------------------------------


def test_a_known_platform_image_needs_no_archive_and_no_build(box):
    box.answer(images=[IMAGE_ID, PLATFORM_REF])
    expect_success(box.run())
    assert not {"ship-platform", "build-platform", "rm-archive", "load"} & set(box.kinds())
    assert box.server1() == ["git toplevel", "git head", "docker inspect", "git archive scripts"]
    assert [c["call"] for c in box.ssh_calls() if c["stdin_size"]] == ["ship-scripts"]
    assert box.remote("has-image") == [f"docker image inspect '{IMAGE_ID}'", f"docker image inspect '{PLATFORM_REF}'"]
    assert box.sleeps() == []


def test_a_missing_platform_image_is_shipped_as_one_file_under_a_temporary_name_and_built(box):
    box.answer(images=[IMAGE_ID])
    expect_success(box.run())
    part, final = f"{box.archive()}.part", box.archive()
    assert box.remote("ship-platform") == [f"cat > '{part}' && mv '{part}' '{final}'"]
    assert box.remote("build-platform") == [box.cmd(f"./stack_up.sh build platform '{COMMIT}' '{final}'")]
    (ship,) = box.of("ship-platform")
    (git,) = box.of("git archive commit")
    assert ship["stdin_sha"] == git["sha"] and git["argv"][1:] == ["-C", str(box.checkout), "archive", "--format=tar", COMMIT]
    kinds = box.kinds()
    assert kinds.index("ship-platform") + 1 == kinds.index("build-platform") and kinds.count("build-platform") == 1


def test_the_wait_checks_at_once_then_every_ten_seconds_and_the_archive_is_removed_when_the_image_is_there(box):
    box.answer(images=[IMAGE_ID], ready_after=420)
    expect_success(box.run())
    kinds = box.kinds()
    assert kinds[kinds.index("build-platform") + 1] == "has-image"
    # the question before the shipping, then the first look at once and one more look after each of the 42 sleeps
    assert box.remote("has-image")[1:] == [f"docker image inspect '{PLATFORM_REF}'"] * 44
    assert box.sleeps() == ["10"] * 42 and box.elapsed() == 420
    assert kinds.index("build-platform") < kinds.index("rm-archive") < kinds.index("build-proxy")
    assert kinds[kinds.index("rm-archive") - 1] == "has-image"
    assert box.remote("rm-archive") == [f"rm '{box.archive()}'"]


def test_the_25_minute_box_ends_the_wait_and_leaves_the_build_and_the_archive_alone(box):
    box.answer(images=[IMAGE_ID], ready_after=None, advance={"build-platform": 123})
    expect_failure(box.run(), "box")
    assert 1500 <= box.elapsed() < 1510 and set(box.sleeps()) == {"10"}
    kinds = box.kinds()
    assert kinds.count("build-platform") == 1 and kinds[-1] == "has-image"
    assert set(kinds) <= {"versions", "ship-scripts", "has-image", "tag", "ship-platform", "build-platform"}
    for call in box.ssh_calls():
        if call["call"] != "ship-scripts":
            assert not re.search(r"\b(kill|stop|restart|rm)\b", call["remote"]), call["remote"]


def test_a_platform_question_that_fails_ends_the_run_before_shipping(box):
    box.answer(images=[IMAGE_ID], rc={f"has-image {PLATFORM_REF}": 2})
    expect_failure(box.run(), "image-ask")
    assert "ship-platform" not in box.kinds()


@pytest.mark.parametrize("rc", [{"ship-platform": 1}, {"build-platform": 1}, {"git archive commit": 1}],
                         ids=["ship", "build", "git"])
def test_a_platform_archive_or_build_that_fails_ends_the_run(box, rc):
    box.answer(images=[IMAGE_ID], rc=rc)
    expect_failure(box.run(), "platform-build")
    assert not {"build-proxy", "rm-archive", "converge"} & set(box.kinds())


# ---- d. proxy build, e. converge -------------------------------------------------------------------


def test_the_full_run_goes_in_the_order_of_the_plan_and_converge_gets_exactly_four_arguments(box):
    expect_success(box.run())
    assert box.kinds() == FULL_RUN
    assert box.remote("build-proxy") == [box.cmd("./stack_up.sh build proxy")]
    assert box.remote("converge") == [box.converge_cmd()]
    assert all(call["stdin_size"] == 0 for call in box.ssh_calls() if call["call"] in ("converge", "build-proxy"))


def test_a_proxy_tag_of_63_plain_characters_is_passed_on_to_converge(box):
    tag = "a" * 30 + "." + "B-c_" * 8
    assert len(tag) == 63
    box.answer(out={"build-proxy": f"stack_up: built proxy {tag}\n"})
    expect_success(box.run())
    assert box.remote("converge") == [box.converge_cmd(tag)]


PROXY_BAD = {
    "no line": ("stack_up: built app 1\n", 0), "empty": ("", 0), "tag not plain": ("stack_up: built proxy a b\n", 0),
    "tag too long": (f"stack_up: built proxy {'a' * 64}\n", 0), "empty tag": ("stack_up: built proxy \n", 0),
    "slash in tag": ("stack_up: built proxy a/b\n", 0), "text before": (f"x stack_up: built proxy {PROXY_TAG}\n", 0),
    "nonzero exit": (f"stack_up: built proxy {PROXY_TAG}\n", 1),
}


@pytest.mark.parametrize("out, rc", list(PROXY_BAD.values()), ids=list(PROXY_BAD))
def test_a_proxy_build_that_reports_no_plain_tag_ends_with_its_reason(box, out, rc):
    box.answer(out={"build-proxy": out}, rc={"build-proxy": rc})
    expect_failure(box.run(), "proxy-tag")
    assert "converge" not in box.kinds()


@pytest.mark.parametrize("out", [
    "stack_up: making\nstack_up: the stack answers\n",
    "stack_up: making\n\nstack_up: the first reset starts the owner app and the gateway\n\n\n",
    "stack_up: the stack answers",
], ids=["answers", "first reset, blank lines", "no final newline"])
def test_both_success_lines_of_converge_are_accepted(box, out):
    box.answer(out={"converge": out})
    expect_success(box.run())


CONVERGE_BAD = {
    "other last line": ("stack_up: the stack answers\nstack_up: waiting\n", 0), "empty output": ("", 0),
    "nonzero exit": ("stack_up: the stack answers\n", 1), "trailing space": ("stack_up: the stack answers \n", 0),
    "prefix only": ("stack_up: the stack\n", 0),
    "failure line": ("stack_up: container x is not one this script made; it was left alone\n", 1),
}


@pytest.mark.parametrize("out, rc", list(CONVERGE_BAD.values()), ids=list(CONVERGE_BAD))
def test_a_converge_that_does_not_end_well_ends_with_its_reason(box, out, rc):
    box.answer(out={"converge": out}, rc={"converge": rc})
    expect_failure(box.run(), "converge")
    assert not {"copy-page-check", "list-tags", "remove-tag"} & set(box.kinds())


# ---- f. the page check copy ------------------------------------------------------------------------


def test_the_page_check_is_copied_by_one_command_after_converge_and_before_the_prune(box):
    expect_success(box.run())
    kinds = box.kinds()
    assert kinds.index("converge") + 1 == kinds.index("copy-page-check") == kinds.index("list-tags") - 1
    (command,) = box.remote("copy-page-check")
    assert f"'{box.app_c}:{OPT}" in command and f"'{box.browser_c}:{BROWSER_PAGE_CHECK}.new'" in command
    assert command.count("docker cp ") == 2 and box.of("copy-page-check")[0]["stdin_size"] == 0


def test_the_copy_replaces_the_old_page_check_and_leaves_no_sibling(box):
    source = tree(box.source_folder)
    for suffix in ("new", "old"):
        stale = box.old_copy.parent / f"page-check.{suffix}"
        stale.mkdir()
        (stale / "junk.txt").write_text("JUNK\n", encoding="utf-8")
    expect_success(box.run())
    assert tree(box.old_copy) == source and ".hidden" in source and "lib/helper.js" in source
    assert [path.name for path in box.old_copy.parent.iterdir()] == ["page-check"]


def test_the_copy_creates_the_folders_when_the_browser_container_has_none(box):
    source = tree(box.source_folder)
    shutil.rmtree(box.fs / box.browser_c)
    (box.fs / box.browser_c).mkdir()
    expect_success(box.run())
    assert tree(box.old_copy) == source


def test_renamed_containers_of_the_template_are_the_ones_used(box):
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


@pytest.mark.parametrize("setup", [
    _no_source, _empty_source, _no_browser, lambda box: box.answer(rc={"s2docker cp-in": 1}),
    lambda box: box.answer(rc={"copy-page-check": 1}),
], ids=["no source", "empty source", "no browser", "copy-in fails", "forced failure"])
def test_a_failed_copy_ends_the_run_keeps_the_old_copy_and_prunes_nothing(box, setup):
    setup(box)
    expect_failure(box.run(), "page-check")
    if (box.fs / box.browser_c).exists():
        assert tree(box.old_copy) == {"old.txt": "OLD\n"}
    assert not {"list-tags", "remove-tag"} & set(box.kinds())


# ---- g. the prune ----------------------------------------------------------------------------------

OLD = {n: str(n) * 40 for n in (1, 2, 3, 4)}
PRUNE_TAGS = {
    "raphael-journeys-proxy": ["a" * 12, "b" * 12, "c" * 12, PROXY_TAG, "d" * 12],
    "raphael-journeys-app-base": [OLD[1], OLD[2], OLD[3], REVISION, OLD[4]],
    "raphael-journeys-app": [f"{OLD[1]}-{CA}", f"{OLD[2]}-{CA}", f"{OLD[3]}-{CA}", APP_TAG, f"{REVISION}-{'b' * 12}", f"{OLD[4]}-{CA}"],
    "raphael-journeys-platform": ["p1" * 20, "p2" * 20, "<none>", "bad tag!", "p3" * 20, "x" * 64, COMMIT, "p4" * 20],
    "raphael-journeys-browser": [f"{OLD[1]}-{CA}", APP_TAG],
}
PRUNED = {
    "raphael-journeys-proxy": ["c" * 12, "d" * 12], "raphael-journeys-app-base": [OLD[3], OLD[4]],
    "raphael-journeys-app": [f"{OLD[3]}-{CA}", f"{OLD[4]}-{CA}"], "raphael-journeys-platform": ["p3" * 20, "p4" * 20],
    "raphael-journeys-browser": [],
}


def test_the_prune_keeps_the_two_newest_and_what_this_run_used_and_removes_the_rest_one_by_one(box):
    box.answer(tags=PRUNE_TAGS)
    expect_success(box.run())
    kinds = box.kinds()
    after = [call["remote"] for call in box.ssh_calls()[kinds.index("copy-page-check") + 1:]]
    expected = []
    for repo in REPOSITORIES:
        expected.append("docker image ls '" + repo + "' --format '{{.Tag}}'")
        expected += [f"docker image rm '{repo}:{tag}'" for tag in PRUNED[repo]]
    assert after == expected
    removed = {call["arg"] for call in box.of("remove-tag")}
    for kept in (f"raphael-journeys-proxy:{PROXY_TAG}", f"raphael-journeys-app-base:{REVISION}", PLATFORM_REF,
                 f"raphael-journeys-app:{APP_TAG}", f"raphael-journeys-app:{REVISION}-{'b' * 12}"):
        assert kept not in removed
    assert not any("<none>" in ref or " " in ref or "x" * 64 in ref for ref in removed)


@pytest.mark.parametrize("count", [0, 1, 2])
def test_a_repository_with_two_tags_or_fewer_loses_nothing(box, count):
    box.answer(tags={repo: [f"{'e' * 11}{i}" for i in range(count)] for repo in REPOSITORIES})
    expect_success(box.run())
    assert box.of("remove-tag") == [] and len(box.of("list-tags")) == 5


@pytest.mark.parametrize("rc, removals, listings", [
    ({"remove-tag raphael-journeys-proxy:" + "c" * 12: 1}, 1, 1),
    # the proxy and app base repositories lost two tags each before the listing of the third failed
    ({"list-tags raphael-journeys-app": 1}, 4, 3),
], ids=["removal", "listing"])
def test_a_failed_removal_or_listing_ends_the_run_with_its_reason(box, rc, removals, listings):
    box.answer(tags=PRUNE_TAGS, rc=rc)
    expect_failure(box.run(), "prune")
    assert len(box.of("remove-tag")) == removals and len(box.of("list-tags")) == listings


# ---- every call is made through the one destination --------------------------------------------------


def test_the_runner_destination_is_the_first_argument_of_every_ssh_call(box):
    expect_success(box.run(JOURNEY_RUNNER_SSH="other_runner@host-2.invalid"))
    calls = box.ssh_calls()
    assert len(calls) == len(FULL_RUN) and {call["dest"] for call in calls} == {"other_runner@host-2.invalid"}
    assert all(call["argv"][1] == call["dest"] and call["nargs"] == 2 for call in calls)


def test_nothing_but_the_three_streams_carries_stdin_and_no_value_travels(box):
    box.answer(tags=PRUNE_TAGS)
    expect_success(box.run())
    assert [c["call"] for c in box.ssh_calls() if c["stdin_size"]] == ["ship-scripts", "load", "ship-platform"]
    assert box.calls() and not any(SECRET in json.dumps(call) for call in box.calls())
