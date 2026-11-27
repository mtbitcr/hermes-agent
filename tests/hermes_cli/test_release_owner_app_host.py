"""Card 4: the owner app container's moves and readbacks, run against a stand-in for docker.

The stand-in is the only docker on PATH, so no test can reach a real container tool. It keeps its
containers in a JSON file and records every call, so each test asserts the exact commands the
adapter issues and the containers they leave behind.
"""

from __future__ import annotations

import csv
import errno
import http.server
import inspect
import json
import logging
import shutil
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

import pytest

from hermes_cli import release_owner_app_host
from hermes_cli.release_host_actions import READBACK_TIMEOUT_SECONDS
from hermes_cli.release_owner_app_host import OwnerAppHost

# Stand-in names and versions, so no call could reach a real container even by mistake.
CONTAINER = "stand-in-workspace"
BACKUP = "stand-in-workspace-rollback"
LABEL = "org.opencontainers.image.revision"
PREV, NEW = "1" * 40, "2" * 40
OLD_IMAGE, NEW_IMAGE = f"stand-in-workspace:{PREV}", f"stand-in-workspace:{NEW}"
# A value in the old container's environment: it goes to the one docker call that makes the new
# container, on that call's standard input, and nowhere else (D-10).
SECRET = "stand-in-secret-value"
HEALTH_BODY = '{"status":"ok"}\n'
# The machine's node, found before the stand-in docker is put alone on PATH: the health read that
# runs inside the container is run with it against a local test server.
NODE = shutil.which("node")


def old_container() -> dict:
    """The running owner app container as docker inspect shows it: every setting of the helper's
    list (B3) and, beside them, what the image and docker set, which the helper leaves to the new
    image: the command, user, host name, labels, log driver and shared memory size."""
    return {
        "Id": "the-old-container",
        "Config": {
            "Image": OLD_IMAGE,
            "Labels": {LABEL: PREV, "org.opencontainers.image.title": "stand-in-workspace"},
            "Env": ["PORT=9121", f"STAND_IN_TOKEN={SECRET}", "PATH=/usr/local/bin:/usr/bin:/bin"],
            "Cmd": ["node", "server.js"],
            "User": "node",
            "Hostname": "stand-in-host",
        },
        "HostConfig": {
            "NetworkMode": "host",
            "RestartPolicy": {"Name": "unless-stopped", "MaximumRetryCount": 0},
            "ReadonlyRootfs": True,
            "CapDrop": ["ALL"],
            "CapAdd": ["NET_BIND_SERVICE"],
            "SecurityOpt": ["no-new-privileges:true"],
            "PidsLimit": 256,
            "Memory": 1073741824,
            "NanoCpus": 1500000000,
            "LogConfig": {"Type": "json-file", "Config": {}},
            "ShmSize": 67108864,
            "AutoRemove": False,
            # The bind mounts as docker run --mount made them; the top-level Mounts show them too.
            "Mounts": [
                {"Type": "bind", "Source": "/srv/stand-in/data", "Target": "/data"},
                {
                    "Type": "bind",
                    "Source": "/srv/stand-in/access",
                    "Target": "/run/access",
                    "ReadOnly": True,
                },
            ],
        },
        "Mounts": [
            {
                "Type": "bind",
                "Source": "/srv/stand-in/data",
                "Destination": "/data",
                "Mode": "",
                "RW": True,
                "Propagation": "rprivate",
            },
            {
                "Type": "bind",
                "Source": "/srv/stand-in/access",
                "Destination": "/run/access",
                "Mode": "ro",
                "RW": False,
                "Propagation": "rprivate",
            },
        ],
        "State": {
            "Status": "running",
            "Running": True,
            "Paused": False,
            "Restarting": False,
            "Dead": False,
            "Health": {"Status": "healthy"},
        },
        "RestartCount": 0,
    }


# The ten names docker's client fills from its own proxy configuration unless docker run gives
# them with --env. docker run gives each without a value, so the client adds none, and takes each
# from the environment it runs with, which holds exactly the old container's value; a name the old
# container leaves unset stays bare, which docker reads as not set (round two, finding 2).
PROXY_NAMES = [
    "HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "NO_PROXY", "no_proxy",
    "FTP_PROXY", "ftp_proxy", "ALL_PROXY", "all_proxy",
]
BARE_PROXIES = [argument for name in PROXY_NAMES for argument in ("--env", name)]
# The one docker run that makes the new container: each setting of the old one, in the helper's
# order, and the new image last. Nothing else of the old container comes along. The environment
# is no argument: docker run reads it as an env file on its standard input (finding 1, D-10), and
# the proxy names come without a value.
RUN_NEW = [
    "run", "-d", "--name", CONTAINER,
    "--network", "host",
    "--restart", "unless-stopped",
    "--read-only",
    "--cap-drop", "ALL",
    "--cap-add", "NET_BIND_SERVICE",
    "--security-opt", "no-new-privileges:true",
    "--pids-limit", "256",
    "--memory", "1073741824",
    "--cpus", "1.5",
    *BARE_PROXIES,
    "--env-file", "/dev/stdin",
    "--mount", "type=bind,src=/srv/stand-in/data,dst=/data",
    "--mount", "type=bind,src=/srv/stand-in/access,dst=/run/access,readonly",
    NEW_IMAGE,
]
# What docker run reads there: the old container's environment, one entry a line.
ENV_LINES = f"PORT=9121\nSTAND_IN_TOKEN={SECRET}\nPATH=/usr/local/bin:/usr/bin:/bin\n"
INSPECT = ["inspect", "--type", "container", CONTAINER]
INSPECT_BACKUP = ["inspect", "--type", "container", BACKUP]
INSPECT_IMAGE = ["inspect", "--type", "image", NEW_IMAGE]
RENAME_TO_BACKUP = ["rename", CONTAINER, BACKUP]
STOP_BACKUP = ["stop", "--time", "30", BACKUP]
REMOVE = ["rm", "-f", CONTAINER]
RENAME_BACK = ["rename", BACKUP, CONTAINER]
START = ["start", CONTAINER]
# The name a move keeps the last working container under, stopped, until the container that takes
# its place passed its health check: docker's health status, then the body read inside it.
KEPT = "stand-in-workspace-last-working"
RENAME_TO_KEPT = ["rename", CONTAINER, KEPT]
STOP_KEPT = ["stop", "--time", "30", KEPT]
RENAME_KEPT_BACK = ["rename", KEPT, CONTAINER]
REMOVE_KEPT = ["rm", "-f", KEPT]
PASSED = [INSPECT, ["exec", CONTAINER, "node", "-e", release_owner_app_host.HEALTH_READ]]
# The forward reads the new image before any change (point 14), and checks the new container.
FORWARD = [INSPECT, INSPECT_IMAGE, RENAME_TO_BACKUP, STOP_BACKUP, RUN_NEW, *PASSED]
# Back reads the backup before it changes anything (finding 2), and removes the new container
# only once the old one passed its health check.
BACK = [INSPECT_BACKUP, RENAME_TO_KEPT, STOP_KEPT, RENAME_BACK, START, *PASSED, REMOVE_KEPT]
FORWARD_REPEAT = [
    INSPECT_BACKUP, INSPECT_IMAGE, RENAME_TO_KEPT, STOP_KEPT, RUN_NEW, *PASSED, REMOVE_KEPT
]
# The restore of the old container while the new one runs under the name.
RESTORE = [
    INSPECT, INSPECT_BACKUP, RENAME_TO_KEPT, STOP_KEPT, RENAME_BACK, START, *PASSED, REMOVE_KEPT
]

# The stand-in docker. It records every call as one JSON line, and what a call reads from the env
# file it names, as docker run does before anything else, as one more line in another file. Then
# it acts on its containers as docker does: a missing container, a name already taken or a command
# named in "fail" is refused, and a refusal writes every argument and the env file to stderr, so a
# test sees whether the adapter passes on what docker writes. docker inspect answers a missing
# container with docker's own words, and "inspect_answers" gives the read of one name another
# answer. A container that docker run makes, or docker start starts, runs healthy with no restart
# counted; one that docker stop stops has exited, and is gone when it was made with auto-remove.
# Each state it sets carries docker's own flags beside the status. With "rm_leaves", docker rm
# fails as a removal docker cannot finish does, leaving the container readable but dead or being
# removed, and docker start refuses such a container. docker run makes the container's environment
# as docker's client does: the env file's entries, then the --env entries, a name without a value
# taking its value from the stand-in's own environment when that sets it, and then each proxy
# setting of the client configuration in "client_proxies" under both cases of its name, unless
# --env gives that name. docker exec answers as the in-container health read does with the status
# and body in "health": it writes nothing and exits 0 only for 200 with exactly the helper's body.
# docker inspect of an image prints the image's configuration in "images", none by default, and
# docker run adds each entry of that image's environment whose name the run gives no entry, as
# docker does. "comes_up" gives the container with an Id the state it reads once docker run made it
# or docker start started it, or "refused", which fails that call and leaves the container as it
# was, or created. A container that reads "removing" has a removal under way: docker rm refuses a
# second one with docker's own words (Moby's daemon/delete.go), and the one under way then ends, the
# container gone, once docker inspect read it "removal_reads" more times, none by default. A
# command in "removal_ends_before" finds such a container gone already, as docker does when the
# removal under way ends just before it.
STAND_IN_DOCKER = r'''
import json
import os
import sys
import time

STATE = os.environ["STAND_IN_DOCKER_STATE"]
args = sys.argv[1:]
with open(os.environ["STAND_IN_DOCKER_CALLS"], "a", encoding="utf-8") as calls:
    calls.write(json.dumps(args) + "\n")
env_file = ""
if "--env-file" in args:
    with open(args[args.index("--env-file") + 1], "rb") as named:
        env_file = named.read().decode("utf-8")
    with open(os.environ["STAND_IN_DOCKER_INPUTS"], "a", encoding="utf-8") as inputs:
        inputs.write(json.dumps(env_file) + "\n")
with open(STATE, encoding="utf-8") as saved:
    state = json.load(saved)
containers = state["containers"]
command, last = args[0], args[-1]
# docker's flags beside the status, as a container that does not run has them.
STOPPED = {"Running": False, "Paused": False, "Restarting": False, "Dead": False}


def save():
    with open(STATE, "w", encoding="utf-8") as saved:
        json.dump(state, saved)


def refuse(code=1):
    sys.stderr.write("stand-in docker refused: " + " ".join(args) + "\n" + env_file)
    sys.exit(code)


def running(name):
    containers[name]["State"] = {
        "Status": "running", **STOPPED, "Running": True, "Health": {"Status": "healthy"}
    }
    containers[name]["RestartCount"] = 0


def comes_up(name, code):
    up = state.get("comes_up", {}).get(containers[name]["Id"])
    if up == "refused":
        save()
        refuse(code)
    running(name)
    if up:
        containers[name]["State"] = up


def removing(name):
    return containers.get(name, {}).get("State", {}).get("Status") == "removing"


if command in state["fail"]:
    refuse()
if command in state.get("removal_ends_before", []) and removing(last):
    del containers[last]
    save()
    sys.stderr.write("Error response from daemon: No such container: " + last + "\n")
    refuse()
if command == "inspect":
    answer = state.get("inspect_answers", {}).get(last)
    if answer:
        code, out, err = answer
        sys.stdout.write(out)
        sys.stderr.write(err)
        sys.exit(code)
    ending = state.get("ending", {})
    if last in ending:
        ending[last] -= 1
        if ending[last] < 0:
            del ending[last]
            del containers[last]
        save()
    if args[2] == "image":
        sys.stdout.write(json.dumps([{"Config": state.get("images", {}).get(last, {})}]) + "\n")
    elif "inspect_prints" in state:
        sys.stdout.write(state["inspect_prints"])
    elif last in containers:
        sys.stdout.write(json.dumps([containers[last]]) + "\n")
    else:
        sys.stdout.write("[]\n")
        sys.stderr.write("Error: No such container: " + last + "\n")
        sys.exit(1)
elif command == "rename":
    if args[1] not in containers or last in containers:
        refuse()
    containers[last] = containers.pop(args[1])
elif command == "stop":
    if last not in containers:
        refuse()
    containers[last]["State"] = {"Status": "exited", **STOPPED, "Health": {"Status": "unhealthy"}}
    if containers[last].get("HostConfig", {}).get("AutoRemove"):
        del containers[last]
elif command == "start":
    if last not in containers or containers[last]["State"].get("Status") in ("dead", "removing"):
        refuse()
    comes_up(last, 1)
elif command == "rm":
    if last not in containers:
        refuse()
    if removing(last):
        sys.stderr.write(
            "Error response from daemon: removal of container " + last + " is already in progress\n"
        )
        if state.get("removal_reads"):
            state.setdefault("ending", {})[last] = state["removal_reads"]
        else:
            del containers[last]
        save()
        refuse()
    if state.get("rm_leaves"):
        left = state["rm_leaves"]
        containers[last]["State"] = {"Status": left, **STOPPED, "Dead": left == "dead"}
        save()
        refuse()
    del containers[last]
elif command == "run":
    name = args[args.index("--name") + 1]
    if name in containers:
        refuse(125)
    given = [args[at + 1] for at, argument in enumerate(args) if argument == "--env"]
    given = [
        entry + "=" + os.environ[entry] if "=" not in entry and entry in os.environ else entry
        for entry in given
    ]
    names = [entry.partition("=")[0] for entry in given]
    for proxy, value in state.get("client_proxies", {}).items():
        for case in (proxy, proxy.lower()):
            if value and case not in names:
                given.append(case + "=" + value)
    env = [line for line in env_file.split("\n") if line] + given
    names = [entry.partition("=")[0] for entry in env]
    image = state.get("images", {}).get(last, {})
    env += [entry for entry in image.get("Env") or [] if entry.partition("=")[0] not in names]
    state["made"] += 1
    containers[name] = {
        "Id": "made-%d" % state["made"],
        "Config": {
            "Image": last,
            "Labels": {state["label"]: state["revisions"][last]},
            "Env": env,
        },
        "State": {"Status": "created", **STOPPED},
        "RestartCount": 0,
    }
    if state.get("run_start_fails"):
        save()
        refuse(125)
    comes_up(name, 125)
elif command == "exec":
    if containers.get(args[1], {}).get("State", {}).get("Status") != "running":
        refuse()
    if state.get("late"):
        time.sleep(60)
    if state["health"] != [200, '{"status":"ok"}\n']:
        sys.exit(1)
else:
    refuse()
save()
'''


class StandInDocker:
    """The stand-in's containers and the calls it saw, read from and written to its files."""

    def __init__(self, directory: Path) -> None:
        self.state_file = directory / "docker-state.json"
        self.calls_file = directory / "docker-calls.jsonl"
        self.inputs_file = directory / "docker-inputs.jsonl"

    def state(self) -> dict:
        return json.loads(self.state_file.read_text(encoding="utf-8"))

    def save(self, state: dict) -> None:
        self.state_file.write_text(json.dumps(state), encoding="utf-8")

    def change(self, **values) -> None:
        self.save({**self.state(), **values})

    def containers(self) -> dict:
        return self.state()["containers"]

    def put(self, name: str, record: dict) -> None:
        state = self.state()
        state["containers"][name] = record
        self.save(state)

    def calls(self) -> list[list[str]]:
        if not self.calls_file.exists():
            return []
        lines = self.calls_file.read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines]

    def inputs(self) -> list[str]:
        """What each call that named an env file read from it, in order."""
        if not self.inputs_file.exists():
            return []
        lines = self.inputs_file.read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines]

    def forget_calls(self) -> None:
        self.calls_file.unlink(missing_ok=True)
        self.inputs_file.unlink(missing_ok=True)


@pytest.fixture(autouse=True)
def docker(tmp_path, monkeypatch) -> StandInDocker:
    """Every test runs with the stand-in docker alone on PATH, holding the old container running
    under its name, so none can reach a real docker, container or image."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "docker").write_text(f"#!{sys.executable}\n{STAND_IN_DOCKER}", encoding="utf-8")
    (bin_dir / "docker").chmod(0o755)
    monkeypatch.setenv("PATH", str(bin_dir))
    stand_in = StandInDocker(tmp_path)
    monkeypatch.setenv("STAND_IN_DOCKER_STATE", str(stand_in.state_file))
    monkeypatch.setenv("STAND_IN_DOCKER_CALLS", str(stand_in.calls_file))
    monkeypatch.setenv("STAND_IN_DOCKER_INPUTS", str(stand_in.inputs_file))
    # A move reads a health check that does not pass again and again; here with no wait between.
    monkeypatch.setattr(release_owner_app_host, "_wait", lambda seconds: None)
    # docker's client launches, but once for each command in "launch_fails": the system refuses
    # that launch, as fork does with EAGAIN when no process can be spared, and docker never sees it.
    real_run = subprocess.run

    def launch(argv, **options):
        state = stand_in.state()
        if argv[:1] == ["docker"] and argv[1] in state.get("launch_fails", []):
            state["launch_fails"].remove(argv[1])
            stand_in.save(state)
            raise BlockingIOError(errno.EAGAIN, "Resource temporarily unavailable")
        return real_run(argv, **options)

    monkeypatch.setattr(subprocess, "run", launch)
    stand_in.save(
        {
            "containers": {CONTAINER: old_container()},
            "revisions": {OLD_IMAGE: PREV, NEW_IMAGE: NEW},
            "label": LABEL,
            "fail": [],
            "health": [200, HEALTH_BODY],
            "made": 0,
        }
    )
    return stand_in


def host() -> OwnerAppHost:
    """A fresh adapter, as a release or a recovery in another process makes one: the adapter
    keeps nothing between calls, so each step may use its own."""
    return OwnerAppHost(container=CONTAINER, label=LABEL)


def the_run(docker: StandInDocker) -> list[str]:
    """The last docker run call; a forward checks the new container's health after it."""
    return [call for call in docker.calls() if call[0] == "run"][-1]


def assert_only_the_old_container_runs(docker: StandInDocker) -> None:
    """Exactly one container is left: the old one itself, under its own name, on the old image,
    running, with its own settings."""
    containers = docker.containers()
    assert list(containers) == [CONTAINER]
    old = containers[CONTAINER]
    assert (old["Id"], old["Config"]["Image"], old["State"]["Status"]) == (
        "the-old-container", OLD_IMAGE, "running",
    )
    assert (old["HostConfig"], old["Config"]["Env"], old["Mounts"]) == (
        old_container()["HostConfig"], old_container()["Config"]["Env"], old_container()["Mounts"],
    )


# Forward.


def test_forward_makes_the_new_container_with_every_setting_of_the_old_one_and_nothing_more(
    docker,
):
    host().forward(NEW_IMAGE)

    # The running container is read, renamed to the backup name and stopped, and the new one is
    # made in one call that carries every setting of the helper's list from the old one and no
    # other: not its command, user, host name, labels, log driver or shared memory size, which
    # the new image and docker set (its labels would give the new container the old revision).
    assert docker.calls() == FORWARD
    containers = docker.containers()
    assert sorted(containers) == [CONTAINER, BACKUP]
    new = containers[CONTAINER]
    assert (new["Id"], new["Config"]["Image"]) == ("made-1", NEW_IMAGE)
    backup = containers[BACKUP]
    assert (backup["Id"], backup["State"]["Status"]) == ("the-old-container", "exited")


def test_forward_adds_no_argument_for_a_setting_the_old_container_leaves_unset(docker):
    """No network mode, the restart policy "no", a writable root, no capabilities, security
    options or limits, no environment and no mounts: the new container gets docker's defaults,
    but for the proxy names, given without a value so that docker's client adds none of its own
    proxy settings: the old container sets none."""
    old = old_container()
    old["HostConfig"] = {
        "NetworkMode": "",
        "RestartPolicy": {"Name": "no", "MaximumRetryCount": 0},
        "ReadonlyRootfs": False,
        "CapDrop": None,
        "CapAdd": None,
        "SecurityOpt": None,
        "PidsLimit": None,
        "Memory": 0,
        "NanoCpus": 0,
    }
    old["Config"]["Env"] = None
    old["Mounts"] = []
    docker.put(CONTAINER, old)

    host().forward(NEW_IMAGE)

    assert the_run(docker) == ["run", "-d", "--name", CONTAINER, *BARE_PROXIES, NEW_IMAGE]


# What a mount other than a bind mount looks like in docker inspect.
OTHER_MOUNTS = {
    "volume": {
        "Type": "volume",
        "Name": "stand-in-cache",
        "Source": "/var/lib/docker/volumes/stand-in-cache/_data",
        "Destination": "/cache",
        "Driver": "local",
        "Mode": "z",
        "RW": True,
        "Propagation": "",
    },
    "tmpfs": {
        "Type": "tmpfs",
        "Source": "",
        "Destination": "/cache",
        "Mode": "",
        "RW": True,
        "Propagation": "",
    },
}


@pytest.mark.parametrize("kind", list(OTHER_MOUNTS))
def test_a_non_bind_mount_refuses_before_any_stop(docker, kind):
    old = old_container()
    old["Mounts"].append(OTHER_MOUNTS[kind])  # after the bind mounts, so the whole list is read
    docker.put(CONTAINER, old)

    with pytest.raises(RuntimeError, match=kind):
        host().forward(NEW_IMAGE)

    # Only the read: no rename, no stop and no new container, so the old one runs on untouched.
    assert docker.calls() == [INSPECT]
    assert docker.containers() == {CONTAINER: old}


def test_forward_refuses_before_any_stop_while_a_backup_is_there(docker):
    """The backup has one fixed name, so that a recovery in another process finds it. While a
    backup is already there, docker refuses the rename and nothing stops."""
    earlier = old_container()
    earlier["Id"] = "an-earlier-backup"
    earlier["State"] = {"Status": "exited"}
    docker.put(BACKUP, earlier)

    with pytest.raises(RuntimeError, match="^docker rename exited 1$"):
        host().forward(NEW_IMAGE)

    assert docker.calls() == [INSPECT, INSPECT_IMAGE, RENAME_TO_BACKUP]
    containers = docker.containers()
    assert (containers[CONTAINER]["Id"], containers[CONTAINER]["State"]["Status"]) == (
        "the-old-container", "running",
    )
    assert containers[BACKUP]["Id"] == "an-earlier-backup"


def assert_no_secret_in(error: BaseException) -> None:
    """D-10: the error names nothing of the environment or of what docker wrote, and carries no
    other error along that could."""
    assert SECRET not in "".join(traceback.format_exception(error))
    assert (error.__cause__, error.__context__) == (None, None)


@pytest.mark.parametrize("refused", [False, True], ids=["made", "refused"])
@pytest.mark.parametrize("move", ["forward", "forward_repeat"])
def test_the_environment_reaches_docker_run_on_its_standard_input_only(
    docker, capfd, caplog, move, refused
):
    """Each entry of the old container's environment is one line of the env file docker run reads
    from its standard input, exactly as the old container holds it. The placeholder is in no
    argument of any call, and in no error, output or log record, also when docker refuses the run
    and writes back its arguments and the env file."""
    caplog.set_level(logging.DEBUG)
    owner_app = host()
    if move == "forward_repeat":
        owner_app.forward(NEW_IMAGE)
        docker.forget_calls()
    if refused:
        docker.change(fail=["run"])
        with pytest.raises(RuntimeError, match="^docker run exited 1$") as error:
            getattr(owner_app, move)(NEW_IMAGE)
        assert_no_secret_in(error.value)
    else:
        getattr(owner_app, move)(NEW_IMAGE)

    calls = docker.calls()
    assert the_run(docker) == RUN_NEW
    assert docker.inputs() == [ENV_LINES]
    assert [argument for call in calls for argument in call if SECRET in argument] == []
    out, err = capfd.readouterr()
    assert SECRET not in out + err + caplog.text


# Environment entries docker's env file reader cannot give back exactly. It reads a line at a
# time, so a line feed in a value starts another entry and a carriage return before one is
# dropped; it drops white space and a byte order mark before a name, skips a line that starts with
# "#", takes the value of a name with no "=" from its own environment, refuses an empty name or
# one with white space in it, reads UTF-8 only, and holds a line of at most 64 KiB with its line
# feed.
UNCARRIED_ENTRIES = {
    "no =": "STAND_IN_TOKEN",
    "an empty name": f"={SECRET}",
    "white space in the name": f"STAND IN TOKEN={SECRET}",
    "white space before the name": f"\tSTAND_IN_TOKEN={SECRET}",
    "a name starting with #": f"#STAND_IN_TOKEN={SECRET}",
    "a byte order mark before the name": f"\ufeffSTAND_IN_TOKEN={SECRET}",
    "a line feed in the value": f"STAND_IN_TOKEN={SECRET}\nPORT=9122",
    "a carriage return in the value": f"STAND_IN_TOKEN={SECRET}\r",
    "a line of 64 KiB": f"STAND_IN_TOKEN={SECRET}".ljust(64 * 1024, "x"),
    "no UTF-8": f"STAND_IN_TOKEN={SECRET}\udcff",
}


@pytest.mark.parametrize("entry", list(UNCARRIED_ENTRIES))
def test_an_environment_entry_docker_cannot_carry_exactly_refuses_before_any_stop(docker, entry):
    old = old_container()
    old["Config"]["Env"].append(UNCARRIED_ENTRIES[entry])
    docker.put(CONTAINER, old)

    with pytest.raises(RuntimeError) as error:
        host().forward(NEW_IMAGE)

    assert docker.calls() == [INSPECT]
    assert docker.containers() == {CONTAINER: old}
    assert_no_secret_in(error.value)


# Docker's client proxy configuration in the stand-in, a value for each of the five settings it
# knows, and a proxy value under each of the ten names in the release process's own environment:
# neither may reach the new container.
CLIENT_PROXIES = {
    "HTTP_PROXY": "http://stand-in-client-proxy.invalid:3128",
    "HTTPS_PROXY": "http://stand-in-client-proxy.invalid:3129",
    "NO_PROXY": "stand-in-client-no-proxy.invalid",
    "FTP_PROXY": "http://stand-in-client-proxy.invalid:3130",
    "ALL_PROXY": "socks5://stand-in-client-proxy.invalid:1080",
}
HOST_PROXY = "http://stand-in-host-proxy.invalid:3128"
# The old container's own proxy settings, put among its other entries. An empty value is a value;
# a bare name is how a container a forward made shows a proxy name it leaves unset.
INSPECTED_PROXIES = {
    "both cases of a name": [
        "HTTP_PROXY=http://stand-in-inspected-proxy.invalid:8080",
        "http_proxy=http://stand-in-inspected-lower-proxy.invalid:8080",
    ],
    "some names": [
        "https_proxy=http://stand-in-inspected-proxy.invalid:8443",
        "NO_PROXY=stand-in-inspected-no-proxy.invalid,127.0.0.1",
        "ALL_PROXY=socks5://stand-in-inspected-proxy.invalid:1080/?with=equals",
        "no_proxy=",
    ],
    "no proxy name": [],
    "bare names a forward leaves": ["HTTP_PROXY", "all_proxy"],
}


@pytest.mark.parametrize("refused", [False, True], ids=["made", "refused"])
@pytest.mark.parametrize("move", ["forward", "forward_repeat"])
@pytest.mark.parametrize("proxies", list(INSPECTED_PROXIES))
def test_the_new_container_gets_exactly_the_old_one_s_proxy_settings_and_no_other(
    docker, monkeypatch, capfd, caplog, proxies, move, refused
):
    """Docker's client gives the new container its own proxy settings under each proxy name that
    docker run does not give with --env, and fills a name given without a value from the
    environment it runs with. docker run gives all ten names without a value and runs with exactly
    the old container's proxy values in place of the release process's own, so the new container
    gets the old one's environment and no proxy setting of the client or of the release host, and
    no value is in any argument, error, output or log (D-10)."""
    caplog.set_level(logging.DEBUG)
    old = old_container()
    old["Config"]["Env"][1:1] = INSPECTED_PROXIES[proxies]
    docker.put(CONTAINER, old)
    docker.change(client_proxies=CLIENT_PROXIES)
    for name in PROXY_NAMES:
        monkeypatch.setenv(name, HOST_PROXY)
    owner_app = host()
    if move == "forward_repeat":
        owner_app.forward(NEW_IMAGE)
        docker.forget_calls()
    told = ""
    if refused:
        docker.change(fail=["run"])
        with pytest.raises(RuntimeError, match="^docker run exited 1$") as error:
            getattr(owner_app, move)(NEW_IMAGE)
        told = "".join(traceback.format_exception(error.value))
    else:
        getattr(owner_app, move)(NEW_IMAGE)
        made = docker.containers()[CONTAINER]["Config"]["Env"]
        # Each proxy name with exactly the old container's value, or bare, as docker shows a name
        # it leaves unset; and the entries with a value are exactly the old container's.
        entries = [entry for entry in old["Config"]["Env"] if "=" in entry]
        inspected = {entry.partition("=")[0]: entry for entry in entries}
        assert sorted(entry for entry in made if entry.partition("=")[0] in PROXY_NAMES) == sorted(
            inspected.get(name, name) for name in PROXY_NAMES
        )
        assert sorted(entry for entry in made if "=" in entry) == sorted(entries)

    calls = docker.calls()
    assert the_run(docker) == RUN_NEW
    assert docker.inputs() == [ENV_LINES]
    given = [call[at + 1] for call in calls for at, word in enumerate(call) if word == "--env"]
    assert given == PROXY_NAMES
    values = [SECRET, HOST_PROXY, *CLIENT_PROXIES.values()]
    values += [entry.partition("=")[2] for entry in INSPECTED_PROXIES[proxies]]
    out, err = capfd.readouterr()
    seen = [argument for call in calls for argument in call] + [told, out, err, caplog.text]
    assert [value for value in values if value and any(value in text for text in seen)] == []


def refuse_before_any_change(docker: StandInDocker, move: str, change) -> BaseException:
    """`move`, one of the two forward paths, run after `change` to the record it reads: the
    running container's for forward, the backup's for forward_repeat. It refuses with only that
    read, every container stays as it was and the one under the name runs on. The refusal is
    returned, for what it names."""
    if move == "forward":
        record, read, name = old_container(), INSPECT, CONTAINER
    else:
        host().forward(NEW_IMAGE)
        record, read, name = docker.containers()[BACKUP], INSPECT_BACKUP, BACKUP
    change(record)
    docker.put(name, record)
    before = docker.containers()
    docker.forget_calls()

    with pytest.raises(RuntimeError) as error:
        getattr(host(), move)(NEW_IMAGE)

    assert docker.calls() == [read]
    assert docker.containers() == before
    assert before[CONTAINER]["State"]["Status"] == "running"
    return error.value


# Proxy settings the environment docker run runs with cannot carry exactly: a name given twice, a
# value with a NUL, which no environment holds, or with what UTF-8 cannot encode, and an entry
# longer than a line of the env file holds.
UNCARRIED_PROXIES = {
    "a name given twice": [
        "HTTP_PROXY=http://stand-in-inspected-proxy.invalid:8080",
        "HTTP_PROXY=http://stand-in-inspected-other-proxy.invalid:8080",
    ],
    "a name given bare and with a value": [
        "https_proxy",
        "https_proxy=http://stand-in-inspected-proxy.invalid:8443",
    ],
    "a NUL in the value": ["http_proxy=http://stand-in-inspected-proxy.invalid\0:8080"],
    "no UTF-8": ["NO_PROXY=stand-in-inspected-no-proxy.invalid\udcff"],
    "a line of 64 KiB": [
        "ALL_PROXY=socks5://stand-in-inspected-proxy.invalid".ljust(64 * 1024, "x")
    ],
}


@pytest.mark.parametrize("move", ["forward", "forward_repeat"])
@pytest.mark.parametrize("proxies", list(UNCARRIED_PROXIES))
def test_a_proxy_setting_docker_run_cannot_carry_exactly_refuses_before_any_change(
    docker, move, proxies
):
    error = refuse_before_any_change(
        docker, move, lambda record: record["Config"]["Env"].extend(UNCARRIED_PROXIES[proxies])
    )

    assert "stand-in-inspected" not in "".join(traceback.format_exception(error))
    assert_no_secret_in(error)


# A custom seccomp profile as docker inspect shows it: docker's client reads the value of a
# seccomp option other than "unconfined" or "builtin" as a file name and keeps the profile's
# compact JSON, which docker run would read as a file name again; and a value like a file name.
CUSTOM_SECCOMP = {
    "a profile's JSON": (
        'seccomp={"defaultAction":"SCMP_ACT_ERRNO",'
        '"syscalls":[{"names":["stand_in_call"],"action":"SCMP_ACT_ALLOW"}]}'
    ),
    "a file name": "seccomp=/srv/stand-in/stand-in-seccomp.json",
}


@pytest.mark.parametrize("move", ["forward", "forward_repeat"])
@pytest.mark.parametrize("profile", list(CUSTOM_SECCOMP))
def test_a_custom_seccomp_profile_refuses_before_any_change(docker, move, profile):
    """docker run would refuse the profile only once the old container has stopped or gone, and
    no weaker profile takes its place: both forward paths refuse it before anything changes,
    naming nothing of the profile."""
    option = CUSTOM_SECCOMP[profile]
    error = refuse_before_any_change(
        docker, move, lambda record: record["HostConfig"]["SecurityOpt"].append(option)
    )

    told = "".join(traceback.format_exception(error))
    parts = [option.partition("=")[2], "SCMP", "stand_in_call", "/srv/", ".json"]
    assert [part for part in parts if part in told] == []


@pytest.mark.parametrize("move", ["forward", "forward_repeat"])
@pytest.mark.parametrize(
    "option", ["seccomp=unconfined", "seccomp:unconfined", "seccomp=builtin", "seccomp:builtin"]
)
def test_a_seccomp_option_naming_docker_s_own_profile_reaches_docker_run_unchanged(
    docker, move, option
):
    """Docker's own profiles, "unconfined" and "builtin", are no file names: docker run takes the
    option as it is, as it takes every other security option."""
    old = old_container()
    old["HostConfig"]["SecurityOpt"].append(option)
    docker.put(CONTAINER, old)
    owner_app = host()
    if move == "forward_repeat":
        owner_app.forward(NEW_IMAGE)

    getattr(owner_app, move)(NEW_IMAGE)

    run = list(RUN_NEW)
    after = run.index("no-new-privileges:true") + 1
    run[after:after] = ["--security-opt", option]
    assert the_run(docker) == run


@pytest.mark.parametrize("kind", ["legacy tmpfs", "tmpfs", "volume"])
def test_a_mount_only_the_configuration_shows_refuses_before_any_stop(docker, kind):
    """docker run --tmpfs keeps its mount in HostConfig.Tmpfs alone, so the top-level Mounts show
    only the two bind mounts; so does any other mount the configuration holds."""
    old = old_container()
    if kind == "legacy tmpfs":
        old["HostConfig"]["Tmpfs"] = {"/cache": "rw,size=64m"}
    else:
        old["HostConfig"]["Mounts"].append({"Type": kind, "Target": "/cache"})
    docker.put(CONTAINER, old)

    with pytest.raises(RuntimeError, match=kind.split()[-1]):
        host().forward(NEW_IMAGE)

    assert docker.calls() == [INSPECT]
    assert docker.containers() == {CONTAINER: old}


def test_a_container_docker_removes_when_it_stops_refuses_before_any_stop(docker):
    """With auto-remove, docker removes the old container when it stops, renamed or not, and no
    backup is left to go back to."""
    old = old_container()
    old["HostConfig"]["AutoRemove"] = True
    docker.put(CONTAINER, old)

    with pytest.raises(RuntimeError):
        host().forward(NEW_IMAGE)

    assert docker.calls() == [INSPECT]
    assert docker.containers() == {CONTAINER: old}
    assert old["State"]["Status"] == "running"


@pytest.mark.parametrize(("retries", "restart"), [(3, "on-failure:3"), (0, "on-failure")])
def test_an_on_failure_restart_keeps_its_retry_count(docker, retries, restart):
    """on-failure:3 restarts the container at most three times; on-failure without a count has
    no limit, as a retry count of 0 reads."""
    old = old_container()
    old["HostConfig"]["RestartPolicy"] = {"Name": "on-failure", "MaximumRetryCount": retries}
    docker.put(CONTAINER, old)

    host().forward(NEW_IMAGE)

    run = list(RUN_NEW)
    run[run.index("unless-stopped")] = restart
    assert the_run(docker) == run


# Bind sources with a comma or a double quote, and the --mount value that carries each. docker
# reads the value as one CSV record, as Go's encoding/csv reads it, so a field with either is
# quoted and a double quote in it doubled.
QUOTED_SOURCES = {
    "/srv/stand-in/a,b": 'type=bind,"src=/srv/stand-in/a,b",dst=/data',
    '/srv/stand-in/"a"': 'type=bind,"src=/srv/stand-in/""a""",dst=/data',
    '/srv/stand-in/a,"b"': 'type=bind,"src=/srv/stand-in/a,""b""",dst=/data',
}


@pytest.mark.parametrize("source", list(QUOTED_SOURCES))
def test_a_bind_source_with_a_comma_or_a_quote_reaches_docker_as_one_field(docker, source):
    old = old_container()
    old["Mounts"][0]["Source"] = source
    old["HostConfig"]["Mounts"][0]["Source"] = source
    docker.put(CONTAINER, old)

    host().forward(NEW_IMAGE)

    run = the_run(docker)
    value = run[run.index("--mount") + 1]
    assert value == QUOTED_SOURCES[source]
    assert next(csv.reader([value], strict=True)) == ["type=bind", f"src={source}", "dst=/data"]


@pytest.mark.parametrize("propagation", ["rslave", "slave", "rshared", "shared", "private"])
def test_a_bind_keeps_a_propagation_other_than_docker_s_default(docker, propagation):
    old = old_container()
    old["Mounts"][0]["Propagation"] = propagation
    old["HostConfig"]["Mounts"][0]["BindOptions"] = {"Propagation": propagation}
    docker.put(CONTAINER, old)

    host().forward(NEW_IMAGE)

    run = list(RUN_NEW)
    run[run.index("type=bind,src=/srv/stand-in/data,dst=/data")] += (
        f",bind-propagation={propagation}"
    )
    assert the_run(docker) == run


# Binds docker run --mount cannot make again as they are, each as a change to what docker inspect
# shows of the first bind mount and to its configuration: an SELinux label, which --mount has no
# field for; a line break in a path, which docker's CSV reader does not give back exactly; and a
# bind option the adapter does not carry over.
UNREPRESENTABLE_BINDS = {
    "a shared SELinux label": ({"Mode": "z"}, {}),
    "a private SELinux label": ({"Mode": "ro,Z", "RW": False}, {}),
    "a line feed in the source": ({"Source": "/srv/stand-in/a\nb"}, {"Source": "/srv/stand-in/a\nb"}),
    "a carriage return in the destination": ({"Destination": "/data\r"}, {"Target": "/data\r"}),
    "a consistency mode": ({"Mode": "cached"}, {}),
    "a consistency setting": ({}, {"Consistency": "cached"}),
    "a non-recursive bind": ({}, {"BindOptions": {"NonRecursive": True}}),
    "a propagation docker does not name": ({"Propagation": "runbindable"}, {}),
}


@pytest.mark.parametrize("bind", list(UNREPRESENTABLE_BINDS))
def test_a_bind_docker_run_cannot_make_again_refuses_before_any_stop(docker, bind):
    shown, configured = UNREPRESENTABLE_BINDS[bind]
    old = old_container()
    old["Mounts"][0].update(shown)
    old["HostConfig"]["Mounts"][0].update(configured)
    docker.put(CONTAINER, old)

    with pytest.raises(RuntimeError):
        host().forward(NEW_IMAGE)

    assert docker.calls() == [INSPECT]
    assert docker.containers() == {CONTAINER: old}


# One setting of each kind a forward refuses, as a change to a container's record.
REFUSED_SETTINGS = {
    "an environment entry": lambda record: record["Config"]["Env"].append("STAND_IN_TOKEN"),
    "a legacy tmpfs mount": lambda record: record["HostConfig"].update(Tmpfs={"/cache": ""}),
    "auto-remove": lambda record: record["HostConfig"].update(AutoRemove=True),
    "an SELinux label": lambda record: record["Mounts"][0].update(Mode="z"),
}


@pytest.mark.parametrize("setting", list(REFUSED_SETTINGS))
def test_forward_repeat_refuses_what_forward_refuses_before_it_removes_anything(docker, setting):
    """Both forwards run the same check: forward_repeat reads the backup, and removes the new
    container only once the backup's settings can be carried over."""
    host().forward(NEW_IMAGE)
    backup = docker.containers()[BACKUP]
    REFUSED_SETTINGS[setting](backup)
    docker.put(BACKUP, backup)
    before = docker.containers()
    docker.forget_calls()

    with pytest.raises(RuntimeError):
        host().forward_repeat(NEW_IMAGE)

    assert docker.calls() == [INSPECT_BACKUP]
    assert docker.containers() == before


# Back, and the helper's stages.


def test_back_puts_the_old_container_back_under_its_own_name(docker):
    host().forward(NEW_IMAGE)
    docker.forget_calls()

    host().back()

    # The backup is read, the new container is renamed and stopped, and the backup is renamed back
    # and started again: the old container itself, not one made anew from its settings. The new
    # container is removed only once the old one passed its health check.
    assert docker.calls() == BACK
    assert_only_the_old_container_runs(docker)


# What docker inspect answers when it reads no container: docker fails to read it, or prints no
# container record. Docker's words carry the placeholder, so a test sees whether they reach an
# error.
FAILED_READS = {
    "a failed read": [1, "", f"Error response from daemon: stand-in failure {SECRET}\n"],
    "an unreadable read": [0, f'[{{"Id": "{SECRET}"', ""],
}


@pytest.mark.parametrize("backup", ["missing", *FAILED_READS])
def test_back_without_a_backup_it_can_read_refuses_before_any_change(docker, backup):
    host().forward(NEW_IMAGE)
    if backup == "missing":
        host().remove_backup()
    else:
        docker.change(inspect_answers={BACKUP: FAILED_READS[backup]})
    before = docker.containers()
    docker.forget_calls()

    with pytest.raises(RuntimeError) as error:
        host().back()

    # Only the read of the backup: the new container runs on, untouched.
    assert docker.calls() == [INSPECT_BACKUP]
    assert docker.containers() == before
    assert before[CONTAINER]["State"]["Status"] == "running"
    assert_no_secret_in(error.value)


def test_the_helpers_stages_end_on_one_new_container_made_from_the_old_ones_settings(
    docker, capfd, caplog
):
    """D-6 and Q5: forward, back, forward, then the helper's forward-repeat, which makes the new
    container once more from the old one's settings, and last the backup is removed."""
    owner_app = host()
    owner_app.forward(NEW_IMAGE)
    owner_app.back()
    owner_app.forward(NEW_IMAGE)
    owner_app.forward_repeat(NEW_IMAGE)
    owner_app.remove_backup()

    assert docker.calls() == [
        *FORWARD,
        *BACK,
        *FORWARD,
        *FORWARD_REPEAT,
        ["rm", "-f", BACKUP],
    ]
    containers = docker.containers()
    assert list(containers) == [CONTAINER]
    assert (containers[CONTAINER]["Id"], containers[CONTAINER]["Config"]["Image"]) == (
        "made-3", NEW_IMAGE,
    )
    # The moves print and log nothing, so the environment they pass to docker goes nowhere else.
    out, err = capfd.readouterr()
    assert (out, err, caplog.text) == ("", "", "")


def test_each_stage_reads_back_the_version_it_put_in_place(docker):
    owner_app = host()
    readbacks = []

    def read_back(expected: str) -> None:
        readbacks.append(
            (
                owner_app.revision_is(expected),
                owner_app.running_unrestarted(),
                owner_app.health_ok(),
            )
        )

    owner_app.forward(NEW_IMAGE)
    read_back(NEW)
    owner_app.back()
    read_back(PREV)
    owner_app.forward(NEW_IMAGE)
    read_back(NEW)
    owner_app.forward_repeat(NEW_IMAGE)
    read_back(NEW)
    owner_app.remove_backup()
    read_back(NEW)

    assert readbacks == [(True, True, True)] * 5


# The restore.

# Each way a forward can stop, with the docker calls the restore then makes. The restore keeps a
# container that runs the old image. Otherwise it reads the backup first, and only once that is
# there on the old image does it rename any other container under the name away and stop it,
# rename the backup back and start it unless it still runs, and remove that other container once
# the backup passed its health check.
FAILED_FORWARDS = {
    # The rename is refused: nothing changed, and the old container still runs under its name.
    "rename": [INSPECT],
    # The stop is refused: the backup still runs, so it is only renamed back.
    "stop": [INSPECT, INSPECT_BACKUP, RENAME_BACK],
    # docker run is refused before it makes anything: the forward itself put the old container
    # back under its name and started it.
    "run": [INSPECT],
    # docker run makes the new container but cannot start it: the forward itself removed that
    # container and put the old one back.
    "run, after making the container": [INSPECT],
    # The forward went through, and the readback after it failed.
    "the readback": RESTORE,
}


@pytest.mark.parametrize("failure", list(FAILED_FORWARDS))
def test_the_restore_after_a_failed_forward_leaves_exactly_one_container_on_the_old_image(
    docker, capfd, caplog, failure
):
    if failure == "the readback":
        host().forward(NEW_IMAGE)
    else:
        if failure == "run, after making the container":
            docker.change(run_start_fails=True)
            refused = "docker run exited 125"
        else:
            docker.change(fail=[failure])
            refused = f"docker {failure} exited 1"
        with pytest.raises(RuntimeError) as error:
            host().forward(NEW_IMAGE)
        # D-10: the error names the docker command and its exit only, never the arguments or what
        # docker wrote, which carry the old container's environment.
        assert str(error.value) == refused
    docker.change(fail=[], run_start_fails=False)
    docker.forget_calls()

    host().restore(OLD_IMAGE)

    assert docker.calls() == FAILED_FORWARDS[failure]
    assert_only_the_old_container_runs(docker)
    out, err = capfd.readouterr()
    assert SECRET not in out + err + caplog.text


@pytest.mark.parametrize(
    ("move", "command"),
    [
        ("back", "rm"),
        ("back", "rename"),
        ("back", "start"),
        ("forward_repeat", "inspect"),
        ("forward_repeat", "rm"),
        ("forward_repeat", "run"),
    ],
)
def test_the_restore_after_a_failed_back_or_forward_repeat_puts_the_old_container_back(
    docker, move, command
):
    """A back whose start is refused, also when it puts the new container back, leaves that one
    stopped under the name; the restore then puts the old container back and starts it, as the
    helper's recovery starts the old container it puts back. A move whose last call, the removal of
    the container it replaced, is refused leaves that one stopped under the name it kept it under,
    and the restore starts from that state as the move left it: after a forward-repeat it removes
    that one once the container under the name passed its health check, before it keeps that
    container there; after a back the old container runs, so it changes nothing."""
    owner_app = host()
    owner_app.forward(NEW_IMAGE)
    if move == "forward_repeat":
        owner_app.back()
        owner_app.forward(NEW_IMAGE)
    docker.change(fail=[command])
    with pytest.raises(RuntimeError, match=f"^docker {command} exited 1$"):
        if move == "back":
            owner_app.back()
        else:
            owner_app.forward_repeat(NEW_IMAGE)
    docker.change(fail=[])
    if command == "rm":
        kept = docker.containers()[KEPT]
        assert (kept["Id"], kept["State"]["Status"]) == (
            "made-1" if move == "back" else "made-2", "exited",
        )

    host().restore(OLD_IMAGE)

    if (move, command) == ("back", "rm"):
        # Left for the next move that keeps a container under that name, as the next test shows.
        state = docker.state()
        kept = state["containers"].pop(KEPT)
        assert (kept["Id"], kept["State"]["Status"]) == ("made-1", "exited")
        docker.save(state)
    assert_only_the_old_container_runs(docker)


# docker inspect of the name a move keeps the last working container under.
INSPECT_KEPT = ["inspect", "--type", "container", KEPT]


def test_a_restore_right_after_a_forward_repeat_whose_last_removal_was_refused_goes_through(docker):
    """A forward-repeat whose removal of the container it replaced docker refused leaves that one
    stopped under the kept name. A restore on a fresh adapter, from that state as the move left
    it, finds the kept name taken before anything stops, reads the one there, and removes it only
    once the container under the name passed its health check, as that one replaced it; then it
    keeps that container there and puts the old one back as always."""
    host().forward(NEW_IMAGE)
    docker.change(fail=["rm"])
    with pytest.raises(RuntimeError, match="^docker rm exited 1$"):
        host().forward_repeat(NEW_IMAGE)
    docker.change(fail=[])
    left = docker.containers()[KEPT]
    assert (left["Id"], left["State"]["Status"]) == ("made-1", "exited")
    docker.forget_calls()

    host().restore(OLD_IMAGE)

    assert docker.calls() == [
        *RESTORE[:3], INSPECT_KEPT, *PASSED, REMOVE_KEPT, *RESTORE[2:]
    ]
    assert_only_the_old_container_runs(docker)


@pytest.mark.parametrize("move", ["back", "restore"])
def test_the_next_rehearsal_after_a_back_or_restore_whose_last_removal_was_refused_goes_through(
    docker, move
):
    """A back or a restore whose removal of the new container docker refused leaves that one
    stopped under the kept name, and the old container running under the name. The next
    rehearsal from that state as the move left it, D-6's forward, back, forward and forward-repeat
    and then a restore, each on a fresh adapter, goes through: its back removes the one left there
    once the container under the name passed its health check, before anything stops."""
    host().forward(NEW_IMAGE)
    docker.change(fail=["rm"])
    with pytest.raises(RuntimeError, match="^docker rm exited 1$"):
        swap(move)
    docker.change(fail=[])
    left = docker.containers()[KEPT]
    assert (left["Id"], left["State"]["Status"]) == ("made-1", "exited")
    docker.forget_calls()

    host().forward(NEW_IMAGE)
    host().back()
    host().forward(NEW_IMAGE)
    host().forward_repeat(NEW_IMAGE)
    host().restore(OLD_IMAGE)

    assert docker.calls() == [
        *FORWARD,
        *BACK[:2], INSPECT_KEPT, *PASSED, REMOVE_KEPT, *BACK[1:],
        *FORWARD,
        *FORWARD_REPEAT,
        *RESTORE,
    ]
    assert_only_the_old_container_runs(docker)


def test_a_restore_with_the_old_container_running_changes_nothing(docker):
    host().restore(OLD_IMAGE)

    assert docker.calls() == [INSPECT]
    assert_only_the_old_container_runs(docker)


@pytest.mark.parametrize("backup", ["missing", *FAILED_READS, "on another image"])
def test_a_restore_without_a_backup_on_the_old_image_leaves_the_current_container_untouched(
    docker, backup
):
    """After a forward the new container runs, so the restore needs the backup. When that is
    missing, cannot be read, or is a container on another image under the backup name, the
    restore refuses and removes nothing."""
    host().forward(NEW_IMAGE)
    if backup == "missing":
        host().remove_backup()
    elif backup == "on another image":
        other = old_container()
        other["Id"] = "another-container"
        other["Config"]["Image"] = "stand-in-other:" + "3" * 40
        docker.put(BACKUP, other)
    else:
        docker.change(inspect_answers={BACKUP: FAILED_READS[backup]})
    before = docker.containers()
    docker.forget_calls()

    with pytest.raises(RuntimeError) as error:
        host().restore(OLD_IMAGE)

    assert docker.calls() == [INSPECT, INSPECT_BACKUP]
    assert docker.containers() == before
    assert_no_secret_in(error.value)


def test_a_restore_with_neither_container_refuses(docker):
    docker.change(containers={})

    with pytest.raises(RuntimeError) as error:
        host().restore(OLD_IMAGE)

    assert docker.calls() == [INSPECT, INSPECT_BACKUP]
    assert docker.containers() == {}
    assert_no_secret_in(error.value)


@pytest.mark.parametrize("words", ["No such container", "No such object"])
def test_a_restore_takes_only_docker_s_word_that_no_container_has_the_name_as_none_there(
    docker, words
):
    """docker inspect exits nonzero and says no container or object has the name: the restore
    then puts the backup back under it."""
    backup = old_container()
    backup["State"] = {"Status": "exited", **STOPPED, "Health": {"Status": "unhealthy"}}
    docker.change(
        containers={BACKUP: backup},
        inspect_answers={CONTAINER: [1, "[]\n", f"Error: {words}: {CONTAINER}\n"]},
    )

    host().restore(OLD_IMAGE)

    assert docker.calls() == [INSPECT, INSPECT_BACKUP, RENAME_BACK, START]
    assert_only_the_old_container_runs(docker)


@pytest.mark.parametrize("read", list(FAILED_READS))
def test_a_restore_that_cannot_read_the_container_under_the_name_refuses_before_any_change(
    docker, read
):
    """Any other failure to read the container under the name leaves it in doubt: the restore
    neither removes it nor puts the backup back."""
    docker.change(run_start_fails=True)
    with pytest.raises(RuntimeError):
        host().forward(NEW_IMAGE)
    docker.change(run_start_fails=False, inspect_answers={CONTAINER: FAILED_READS[read]})
    before = docker.containers()
    docker.forget_calls()

    with pytest.raises(RuntimeError) as error:
        host().restore(OLD_IMAGE)

    assert docker.calls() == [INSPECT]
    assert docker.containers() == before
    assert_no_secret_in(error.value)


# A backup docker cannot put back. The moves that remove the container under the name while the
# backup is the only way back, each with the reads it makes before it would. The backups a forward
# leaves still go back: exited (back and the stages above, and the restore after a refused run)
# and running after a refused stop (the restore, without a start).
LEANING_ON_THE_BACKUP = {
    "back": (lambda owner_app: owner_app.back(), [INSPECT_BACKUP]),
    "restore": (lambda owner_app: owner_app.restore(OLD_IMAGE), [INSPECT, INSPECT_BACKUP]),
    "forward_repeat": (lambda owner_app: owner_app.forward_repeat(NEW_IMAGE), [INSPECT_BACKUP]),
}


@pytest.mark.parametrize("move", list(LEANING_ON_THE_BACKUP))
@pytest.mark.parametrize("left", ["dead", "removing"])
def test_a_failed_backup_removal_leaves_the_app_running_and_no_move_leans_on_that_backup(
    docker, left, move
):
    """docker rm can fail and leave the backup readable but dead, or still being removed, and
    docker starts neither again. The new container runs on, on its image; back, the restore and
    forward_repeat, each of which would remove it, read the backup and refuse before any removal,
    rename, start or run."""
    host().forward(NEW_IMAGE)
    docker.change(rm_leaves=left)
    with pytest.raises(RuntimeError, match="^docker rm exited 1$"):
        host().remove_backup()
    docker.change(rm_leaves=None)
    before = docker.containers()
    current = before[CONTAINER]
    assert (current["Id"], current["Config"]["Image"], current["State"]["Status"]) == (
        "made-1", NEW_IMAGE, "running",
    )
    assert before[BACKUP]["State"]["Status"] == left
    docker.forget_calls()
    lean, reads = LEANING_ON_THE_BACKUP[move]

    with pytest.raises(RuntimeError) as error:
        lean(host())

    assert docker.calls() == reads
    assert docker.containers() == before
    assert_no_secret_in(error.value)


# Backup states other than exited or running, as the backup's State, and states whose flags are
# missing, are not true or false, or do not agree with the status: docker would not start the
# backup again as it ran, or its state is in doubt.
STOPPED = {"Running": False, "Paused": False, "Restarting": False, "Dead": False}
UNUSABLE_STATES = {
    "paused": {"Status": "paused", **STOPPED, "Running": True, "Paused": True},
    "restarting": {"Status": "restarting", **STOPPED, "Running": True, "Restarting": True},
    "created": {"Status": "created", **STOPPED},
    "an unknown status": {"Status": "stand-in-status", **STOPPED},
    "no status": dict(STOPPED),
    "a status that is no string": {"Status": ["exited"], **STOPPED},
    "no Running flag": {"Status": "exited", "Paused": False, "Restarting": False, "Dead": False},
    "a flag that is no boolean": {"Status": "running", **STOPPED, "Running": 1},
    "exited, but Running": {"Status": "exited", **STOPPED, "Running": True},
    "running, but not Running": {"Status": "running", **STOPPED},
    "exited, but Dead": {"Status": "exited", **STOPPED, "Dead": True},
    "running, but Paused": {"Status": "running", **STOPPED, "Running": True, "Paused": True},
    "running, but Restarting": {
        "Status": "running", **STOPPED, "Running": True, "Restarting": True
    },
    # Point 13: docker's words on why the backup did not start; no error may hold them.
    "exited, with a start error": {
        "Status": "exited", **STOPPED, "ExitCode": 127, "Error": f"stand-in start error {SECRET}"
    },
}


@pytest.mark.parametrize("move", list(LEANING_ON_THE_BACKUP))
@pytest.mark.parametrize("state", list(UNUSABLE_STATES))
def test_no_move_leans_on_a_backup_in_any_other_state(docker, state, move):
    host().forward(NEW_IMAGE)
    backup = docker.containers()[BACKUP]
    backup["State"] = UNUSABLE_STATES[state]
    docker.put(BACKUP, backup)
    before = docker.containers()
    docker.forget_calls()
    lean, reads = LEANING_ON_THE_BACKUP[move]

    with pytest.raises(RuntimeError) as error:
        lean(host())

    assert docker.calls() == reads
    assert docker.containers() == before
    assert before[CONTAINER]["State"]["Status"] == "running"
    assert_no_secret_in(error.value)


# The last working container and the one that takes its place, and the move.


# The four moves, each but the forward run after a forward: the Ids of the last working container
# and of the container that takes its place.
SWAPS = {
    "forward": ("the-old-container", "made-1"),
    "back": ("made-1", "the-old-container"),
    "forward_repeat": ("made-1", "made-2"),
    "restore": ("made-1", "the-old-container"),
}


def swap(move: str) -> None:
    owner_app = host()
    if move == "forward":
        owner_app.forward(NEW_IMAGE)
    elif move == "back":
        owner_app.back()
    elif move == "forward_repeat":
        owner_app.forward_repeat(NEW_IMAGE)
    else:
        owner_app.restore(OLD_IMAGE)


# How the container that takes the place fails, as the state docker shows once docker run made it
# or docker start started it, or as a refused run or start (point 19).
TAKING_THE_PLACE_FAILS = {
    "marked dead": {"Status": "dead", **STOPPED, "Dead": True},
    "marked removing": {"Status": "removing", **STOPPED},
    "exited with a start error": {
        "Status": "exited", **STOPPED, "ExitCode": 127, "Error": f"stand-in start error {SECRET}"
    },
    "a refused start": "refused",
    "a failed health check": {
        "Status": "running", **STOPPED, "Running": True, "Health": {"Status": "unhealthy"}
    },
}


@pytest.mark.parametrize("failure", list(TAKING_THE_PLACE_FAILS))
@pytest.mark.parametrize("move", list(SWAPS))
def test_a_move_whose_new_container_fails_puts_the_last_working_one_back_running(
    docker, capfd, caplog, move, failure
):
    """Points 10 and 19: the move takes the container that failed away, gives the last working
    container its name back, starts it, checks its health and reports that it put it back. No
    call removed the last working container: each container is back under its name."""
    caplog.set_level(logging.DEBUG)
    if move != "forward":
        host().forward(NEW_IMAGE)
    last, taking = SWAPS[move]
    before = {name: container["Id"] for name, container in docker.containers().items()}
    docker.change(comes_up={taking: TAKING_THE_PLACE_FAILS[failure]})
    docker.forget_calls()

    with pytest.raises(release_owner_app_host.PutBack) as error:
        swap(move)

    calls = docker.calls()
    made = move in ("forward", "forward_repeat")
    assert [call for call in calls if call[0] == "rm"] == ([REMOVE] if made else [])
    assert calls[-4:] == [RENAME_BACK if move == "forward" else RENAME_KEPT_BACK, START, *PASSED]
    containers = docker.containers()
    assert {name: container["Id"] for name, container in containers.items()} == before
    assert (containers[CONTAINER]["Id"], containers[CONTAINER]["State"]["Status"]) == (
        last, "running",
    )
    assert_no_secret_in(error.value)
    out, err = capfd.readouterr()
    assert SECRET not in out + err + caplog.text


# The call of each move whose launch of docker's client the system refuses once: docker run, which
# makes the container that takes the place, or docker start, which starts the backup in its place.
UNLAUNCHED = {"forward": "run", "back": "start", "forward_repeat": "run", "restore": "start"}


@pytest.mark.parametrize("move", list(SWAPS))
def test_a_docker_client_launch_refused_once_puts_the_last_working_one_back_running(
    docker, capfd, caplog, move
):
    """The system refuses the launch of docker's client once, after the move stopped the last
    working container (BlockingIOError, EAGAIN), and docker answers the next call. That takes the
    way of a refused docker call: when the move reports PutBack, the last working container runs
    under the name again and passed its health check after its start, and no call removed it."""
    caplog.set_level(logging.DEBUG)
    if move != "forward":
        host().forward(NEW_IMAGE)
    last = SWAPS[move][0]
    before = {name: container["Id"] for name, container in docker.containers().items()}
    command = UNLAUNCHED[move]
    docker.change(launch_fails=[command])
    docker.forget_calls()

    with pytest.raises(release_owner_app_host.PutBack) as error:
        swap(move)

    assert str(error.value) == f"docker {command} did not launch (errno {errno.EAGAIN})"
    assert docker.state()["launch_fails"] == []
    calls = docker.calls()
    assert [call for call in calls if call[0] == "rm"] == []
    assert calls[-3:] == [START, *PASSED]
    containers = docker.containers()
    assert {name: container["Id"] for name, container in containers.items()} == before
    assert (containers[CONTAINER]["Id"], containers[CONTAINER]["State"]["Status"]) == (
        last, "running",
    )
    assert host().health_ok() is True
    assert_no_secret_in(error.value)
    out, err = capfd.readouterr()
    assert SECRET not in out + err + caplog.text


# How docker answers the move that takes the container that failed off the name, before the last
# working container gets the name back, each with the moves it meets in and what the container that
# failed reads once docker run made it or docker start started it: a second removal refused while
# docker has one under way, which then ends; the container gone between docker inspect and docker
# stop, as the removal docker had under way ended; a removal refused that leaves the container dead.
DEAD = {"Status": "dead", **STOPPED, "Dead": True}
REMOVING = {"Status": "removing", **STOPPED}
CLEARING_ANSWERS = {
    "a second removal refused while one is under way": (["forward", "forward_repeat"], REMOVING, {}),
    "gone between inspect and stop": (
        ["back", "restore"], REMOVING, {"removal_ends_before": ["stop"]}
    ),
    "dead, its removal refused": (["forward", "forward_repeat"], DEAD, {"rm_leaves": "dead"}),
}


@pytest.mark.parametrize(
    ("move", "answer"),
    [(move, answer) for answer, (moves, _, _) in CLEARING_ANSWERS.items() for move in moves],
)
def test_the_name_is_freed_of_the_container_that_failed_however_docker_answers(
    docker, capfd, caplog, move, answer
):
    """After a refusal, the move reads the name again and goes on once no container has it, and a
    container it made that docker does not remove it renames away, with its Id after the name. The
    last working container then runs under the name and passed its health check, the move reports
    PutBack, and no call removed it: each removal named the container that failed, under the
    name, before the last working container got the name back."""
    caplog.set_level(logging.DEBUG)
    if move != "forward":
        host().forward(NEW_IMAGE)
    last, taking = SWAPS[move]
    _, comes_up, answers = CLEARING_ANSWERS[answer]
    docker.change(comes_up={taking: comes_up}, **answers)
    docker.forget_calls()

    with pytest.raises(release_owner_app_host.PutBack) as error:
        swap(move)

    calls = docker.calls()
    rename_back = RENAME_BACK if move == "forward" else RENAME_KEPT_BACK
    assert calls[-4:] == [rename_back, START, *PASSED]
    made = move in ("forward", "forward_repeat")
    assert [call for call in calls if call[0] == "rm"] == ([REMOVE] if made else [])
    assert calls.index(rename_back) > max(
        [at for at, call in enumerate(calls) if call[0] == "rm"], default=-1
    )
    containers = docker.containers()
    assert (containers[CONTAINER]["Id"], containers[CONTAINER]["State"]["Status"]) == (
        last, "running",
    )
    assert host().health_ok() is True
    failed = [name for name, container in containers.items() if container["Id"] == taking]
    assert failed == ([f"{CONTAINER}-unremoved-{taking}"] if comes_up is DEAD else [])
    assert_no_secret_in(error.value)
    out, err = capfd.readouterr()
    assert SECRET not in out + err + caplog.text


@pytest.mark.parametrize("move", ["forward", "forward_repeat"])
@pytest.mark.parametrize("reads", [2, 1000], ids=["ends", "never ends"])
def test_a_removal_docker_has_under_way_is_waited_for_a_bounded_number_of_times(
    docker, monkeypatch, move, reads
):
    """While docker has a removal of the container that failed under way, the move reads the name
    again, the runner's readback interval apart (`_wait`), at most the runner's number of readback
    attempts. A removal that ends in that time frees the name; one that does not leaves the move to
    raise a plain error, not PutBack, with the last working container kept, stopped, and not
    removed."""
    waits = []
    monkeypatch.setattr(release_owner_app_host, "_wait", waits.append)
    if move != "forward":
        host().forward(NEW_IMAGE)
    last, taking = SWAPS[move]
    docker.change(comes_up={taking: REMOVING}, removal_reads=reads)
    docker.forget_calls()
    waits.clear()

    with pytest.raises(RuntimeError) as error:
        swap(move)

    calls = docker.calls()
    reread = calls[calls.index(REMOVE) + 1 :]
    attempts = release_owner_app_host.READBACK_ATTEMPTS
    assert set(waits) == {release_owner_app_host.READBACK_INTERVAL_SECONDS}
    containers = docker.containers()
    if reads == 2:
        assert type(error.value) is release_owner_app_host.PutBack
        rename_back = RENAME_BACK if move == "forward" else RENAME_KEPT_BACK
        assert reread[:4] == [INSPECT, INSPECT, INSPECT, rename_back]
        assert (containers[CONTAINER]["Id"], containers[CONTAINER]["State"]["Status"]) == (
            last, "running",
        )
    else:
        assert type(error.value) is RuntimeError
        assert str(error.value) == "the name is still taken, so the last working container is not back"
        assert reread == [INSPECT] * (attempts - 1)
        assert len(waits) == 2 * (attempts - 1)
        kept = containers[BACKUP if move == "forward" else KEPT]
        assert (kept["Id"], kept["State"]["Status"]) == (last, "exited")
        assert_no_secret_in(error.value)


@pytest.mark.parametrize("move", list(SWAPS))
def test_a_move_removes_the_last_working_container_only_after_its_health_check_passed(
    docker, move
):
    """Point 19: with no failure, the call that removes the last working container comes after the
    health check the container that took its place passed; before it, the move only renamed and
    stopped it. The forward keeps it as the backup, which remove_backup removes after it."""
    if move != "forward":
        host().forward(NEW_IMAGE)
    last, taking = SWAPS[move]
    docker.forget_calls()

    swap(move)
    if move == "forward":
        host().remove_backup()

    calls = docker.calls()
    removal = ["rm", "-f", BACKUP if move == "forward" else KEPT]
    assert calls == {
        "forward": [*FORWARD, removal],
        "back": BACK,
        "forward_repeat": FORWARD_REPEAT,
        "restore": RESTORE,
    }[move]
    assert calls[calls.index(removal) - 2 : calls.index(removal)] == PASSED
    containers = docker.containers()
    assert (containers[CONTAINER]["Id"], containers[CONTAINER]["State"]["Status"]) == (
        taking, "running",
    )
    assert last not in [container["Id"] for container in containers.values()]


# Volumes a new image declares, by the path docker cleans: the old container binds /data and
# /run/access, and docker makes a volume for each path where no mount is.
UNCOVERED_VOLUMES = {
    "a path no bind is at": {"/data": {}, "/cache": {}},
    "a path under a bind": {"/data/cache": {}},
}


@pytest.mark.parametrize("move", ["forward", "forward_repeat"])
@pytest.mark.parametrize("volumes", list(UNCOVERED_VOLUMES))
def test_a_volume_of_the_new_image_no_bind_covers_refuses_before_any_stop(docker, move, volumes):
    """Point 14: docker run would give the new container an empty volume there. The move reads the
    new image and refuses before any rename or stop; the container under the name runs on."""
    if move == "forward_repeat":
        host().forward(NEW_IMAGE)
    docker.change(images={NEW_IMAGE: {"Volumes": UNCOVERED_VOLUMES[volumes]}})
    before = docker.containers()
    docker.forget_calls()

    with pytest.raises(RuntimeError):
        getattr(host(), move)(NEW_IMAGE)

    assert docker.calls() == [INSPECT if move == "forward" else INSPECT_BACKUP, INSPECT_IMAGE]
    assert docker.containers() == before
    assert before[CONTAINER]["State"]["Status"] == "running"


@pytest.mark.parametrize("move", ["forward", "forward_repeat"])
def test_a_volume_of_the_new_image_at_a_bind_s_path_makes_no_refusal(docker, move):
    owner_app = host()
    if move == "forward_repeat":
        owner_app.forward(NEW_IMAGE)
    docker.change(images={NEW_IMAGE: {"Volumes": {"/data/": {}, "//run/./access": {}}}})

    getattr(owner_app, move)(NEW_IMAGE)

    assert the_run(docker) == RUN_NEW


@pytest.mark.parametrize("bare", [[], ["https_proxy"]], ids=["left out", "bare"])
@pytest.mark.parametrize("move", ["forward", "forward_repeat"])
def test_a_variable_only_the_new_image_declares_comes_from_the_new_image(docker, move, bare):
    """Point 15: as with the helper of today, the new container holds each entry of the old one's
    environment, and each variable only the new image declares, a proxy name too, with the image's
    value; a variable the old container holds keeps the old container's value. A proxy name the old
    container shows bare, as a forward leaves it, gives no value. No value is in an argument."""
    old = old_container()
    old["Config"]["Env"] += ["HTTP_PROXY=http://stand-in-inspected-proxy.invalid:8080", *bare]
    docker.put(CONTAINER, old)
    only_the_image = [
        "STAND_IN_IMAGE_ONLY=stand-in-image-value",
        "https_proxy=http://stand-in-image-proxy.invalid:8443",
    ]
    image_env = [
        "PATH=/stand-in-image/bin",
        "HTTP_PROXY=http://stand-in-image-proxy.invalid:8080",
        *only_the_image,
    ]
    docker.change(images={NEW_IMAGE: {"Env": image_env}}, client_proxies=CLIENT_PROXIES)
    owner_app = host()
    if move == "forward_repeat":
        owner_app.forward(NEW_IMAGE)

    getattr(owner_app, move)(NEW_IMAGE)

    made = docker.containers()[CONTAINER]["Config"]["Env"]
    assert sorted(entry for entry in made if "=" in entry) == sorted(
        [entry for entry in old["Config"]["Env"] if "=" in entry] + only_the_image
    )
    assert the_run(docker) == RUN_NEW
    arguments = [argument for call in docker.calls() for argument in call]
    assert [argument for argument in arguments if "stand-in-image" in argument] == []


# The readbacks.


def test_r1_app_reads_the_revision_label_of_the_container_under_its_name(docker):
    assert (host().revision_is(PREV), host().revision_is(NEW)) == (True, False)

    host().forward(NEW_IMAGE)

    assert (host().revision_is(PREV), host().revision_is(NEW)) == (False, True)


@pytest.mark.parametrize(
    ("status", "restarts", "expected"),
    [
        ("running", 0, True),
        ("running", 2, False),  # restarted twice by its restart policy since it was started
        ("restarting", 1, False),
        ("exited", 0, False),
        ("created", 0, False),
    ],
)
def test_r3_app_reads_the_container_running_with_no_restart_since_its_start(
    docker, status, restarts, expected
):
    old = old_container()
    old["State"]["Status"] = status
    old["RestartCount"] = restarts
    docker.put(CONTAINER, old)

    assert host().running_unrestarted() is expected


def test_r4_app_reads_docker_health_and_the_exact_body_inside_the_container(docker):
    assert host().health_ok() is True

    # Docker's own health check, then /healthz read by node inside the container: no address
    # leaves the container, and none is a setting of the adapter.
    assert docker.calls() == [
        INSPECT,
        ["exec", CONTAINER, "node", "-e", release_owner_app_host.HEALTH_READ],
    ]


# The stand-in's docker exec exits as the read does for each answer; the read itself is run
# against a local test server further down.
@pytest.mark.parametrize(
    "body",
    [
        '{"status":"ok"}',
        '{"status":"ok"}\r\n',
        '{"status":"ok"}\n\n',
        ' {"status":"ok"}\n',
        '{"status": "ok"}\n',
        '{"status":"OK"}\n',
        '{"status":"degraded"}\n',
        "ok\n",
        "",
    ],
)
def test_r4_app_is_false_for_a_200_with_any_other_body(docker, body):
    docker.change(health=[200, body])

    assert host().health_ok() is False


@pytest.mark.parametrize("status", [201, 204, 301, 500, 503])
def test_r4_app_is_false_for_any_other_status_with_the_exact_body(docker, status):
    docker.change(health=[status, HEALTH_BODY])

    assert host().health_ok() is False


@pytest.mark.parametrize("health", ["starting", "unhealthy", "no health check"])
def test_r4_app_is_false_unless_docker_health_reads_healthy(docker, health):
    old = old_container()
    if health == "no health check":
        del old["State"]["Health"]
    else:
        old["State"]["Health"] = {"Status": health}
    docker.put(CONTAINER, old)

    assert host().health_ok() is False


# What a local test server answers on /healthz: the status, the body, and whether R4-app's read
# then exits 0. The redirect leads to a page with the exact body; the late answer comes after the
# read's own time limit, which the test shortens to half a second.
HEALTH_ANSWERS = {
    "200 with the exact body": (200, HEALTH_BODY, True),
    "200 with another body": (200, '{"status":"degraded"}\n', False),
    "200 with the placeholder": (200, f'{{"status":"ok","token":"{SECRET}"}}\n', False),
    "500 with the exact body": (500, HEALTH_BODY, False),
    "503 with the exact body": (503, HEALTH_BODY, False),
    "a redirect to the exact body": (302, HEALTH_BODY, False),
    "an answer after the time limit": (None, HEALTH_BODY, False),
}


@pytest.mark.skipif(NODE is None, reason="node is not on PATH")
@pytest.mark.parametrize("answer", list(HEALTH_ANSWERS))
def test_r4_app_s_read_compares_inside_the_container_and_writes_nothing(answer):
    """The read itself, run by the machine's node against a local test server: it exits 0 only
    for a 200 with exactly the helper's body, and writes nothing, so no body leaves the container
    whatever the route answers."""
    status, body, healthy = HEALTH_ANSWERS[answer]
    released = threading.Event()

    class Healthz(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if status is None:
                released.wait(10)  # until the read has given up at its own limit
                return
            code = status if self.path == "/healthz" else 200
            data = body.encode()
            self.send_response(code)
            if code == 302:
                self.send_header("Location", "/elsewhere")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args) -> None:
            pass

    script = release_owner_app_host.HEALTH_READ
    if status is None:
        limit = f"process.exit(1), {READBACK_TIMEOUT_SECONDS * 1000:.0f})"
        assert limit in script
        script = script.replace(limit, "process.exit(1), 500)")
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Healthz)
    serving = threading.Thread(target=server.serve_forever)
    serving.start()
    try:
        started = time.monotonic()
        done = subprocess.run(
            [NODE, "-e", script],
            env={"PORT": str(server.server_address[1])},
            capture_output=True,
            timeout=30,
            check=False,
        )
        took = time.monotonic() - started
    finally:
        released.set()
        server.shutdown()
        server.server_close()
        serving.join()

    assert (done.returncode == 0) is healthy
    assert (done.stdout, done.stderr) == (b"", b"")
    assert took < 5


# What docker inspect prints in place of a container record.
NOT_A_RECORD = {
    "an empty list": "[]\n",
    "not JSON": "not json\n",
    "no state or settings": '[{"Id": "the-old-container"}]\n',
    "null state and settings": '[{"State": null, "Config": null}]\n',
    "a record not in a list": '{"State": {"Status": "running"}}\n',
}


@pytest.mark.parametrize(
    "case", ["no container", "docker refuses", "no docker", *NOT_A_RECORD]
)
def test_the_readbacks_answer_false_rather_than_raise(docker, tmp_path, monkeypatch, case):
    if case == "no container":
        docker.change(containers={})
    elif case == "docker refuses":
        docker.change(fail=["inspect", "exec"])
    elif case == "no docker":
        (tmp_path / "empty").mkdir()
        monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    else:
        docker.change(inspect_prints=NOT_A_RECORD[case])
    owner_app = host()

    assert owner_app.revision_is(PREV) is False
    assert owner_app.running_unrestarted() is False
    assert owner_app.health_ok() is False


def test_r1_app_is_false_without_the_label(docker):
    old = old_container()
    del old["Config"]["Labels"][LABEL]
    docker.put(CONTAINER, old)

    assert host().revision_is(PREV) is False


def test_every_readback_call_ends_at_the_readback_time_limit(docker, monkeypatch):
    """The readbacks share the release host's time limit, R6's own, and the read inside the
    container ends itself at the same limit, as a docker exec killed at the limit would leave it
    running in the container."""
    limits = []
    real_run = subprocess.run

    def run(argv, **options):
        limits.append((argv[1], options.get("timeout")))
        return real_run(argv, **options)

    monkeypatch.setattr(subprocess, "run", run)
    owner_app = host()
    owner_app.revision_is(PREV)
    owner_app.running_unrestarted()
    owner_app.health_ok()

    limit = READBACK_TIMEOUT_SECONDS
    assert limits == [("inspect", limit), ("inspect", limit), ("inspect", limit), ("exec", limit)]
    assert f"setTimeout(() => process.exit(1), {limit * 1000:.0f})" in (
        release_owner_app_host.HEALTH_READ
    )


def test_a_late_health_answer_reads_false_at_the_time_limit(docker, monkeypatch):
    monkeypatch.setattr(release_owner_app_host, "READBACK_TIMEOUT_SECONDS", 2.0)
    docker.change(late=True)

    started = time.monotonic()
    assert host().health_ok() is False
    assert time.monotonic() - started < 30  # the stand-in would answer only after a minute


# The adapter itself.


def test_the_adapter_has_no_network_address_setting():
    """Acceptance: the only settings are the container's name and the revision label; health is
    read inside the container, at the container's own PORT on its own loopback."""
    assert list(inspect.signature(OwnerAppHost).parameters) == ["container", "label"]
    assert "http://127.0.0.1:${process.env.PORT}/healthz" in release_owner_app_host.HEALTH_READ


def test_the_moves_and_readbacks_import_nothing(docker):
    """K6 (D-7): every import happens when the module loads, never inside a move, a readback or
    the restore."""
    owner_app = host()
    loaded = set(sys.modules)

    owner_app.forward(NEW_IMAGE)
    owner_app.revision_is(NEW)
    owner_app.running_unrestarted()
    owner_app.health_ok()
    owner_app.back()
    owner_app.forward(NEW_IMAGE)
    docker.change(fail=["run"])
    with pytest.raises(RuntimeError):
        owner_app.forward_repeat(NEW_IMAGE)
    docker.change(fail=[])
    owner_app.restore(OLD_IMAGE)
    owner_app.forward(NEW_IMAGE)
    owner_app.forward_repeat(NEW_IMAGE)
    owner_app.remove_backup()

    assert set(sys.modules) - loaded == set()
    assert docker.containers()[CONTAINER]["Config"]["Image"] == NEW_IMAGE
