"""The owner app half of the release host: the owner app container's moves and its readbacks.

The owner app's deploy helper (S3, deploy_workspace_container.py) ported into a host adapter. A
forward renames the running container to the backup name and stops it, then makes the new
container from the running one's own settings, the helper's list (B3) and nothing more: network
mode, restart policy, read-only root, dropped and added capabilities, security options, process
limit, memory, processor share, environment and bind mounts. A mount of any other kind refuses
before anything stops, as do auto-remove (docker would remove the old container when it stops), a
bind mount docker run --mount cannot make again, a custom seccomp profile (docker run would read
the profile docker inspect shows as a file name) and an environment entry docker cannot carry
exactly, and so does a volume the new image declares where no bind mount of the old container is.
Back, once it has read the backup, starts the backup again under its own name, forward-repeat
makes the new container once more from the settings the backup keeps, and the restore takes the
helper's recovery steps, changing nothing before it knows the backup it puts back. Each of the
three renames nothing before the backup reads as one docker can put back as it was: exited, as a
forward's stop leaves it, or running, as a forward whose stop failed leaves it, with no start
error; never dead or being removed, as a failed removal of the backup can leave it.

No move removes the container that ran last before the container that takes its place passed its
health check: until then the move only stops it and renames it, to the backup name in a forward
and to the kept name in the others, so that one container runs at a time. When the container that
takes its place does not start or does not pass its health check, the move takes that one away,
gives the last working container its name back, starts it, checks its health and raises
`PutBack`. Taking it away is safe to retry: the move reads the name again after a refusal, waits a
bounded time for a removal docker has under way, and renames away one docker does not remove.
Which move runs when, the readback after each and when the backup goes belong to the
runner of the owner app part.

Every docker call is one argument list through `_docker`, and no value of the environment is ever
an argument (D-10). Docker run reads the environment on its standard input, as an env file, but
for the proxy settings: it gives each proxy name with --env, without a value, so that docker's
client adds no proxy setting of its own, and takes the old container's values from the
environment it runs with, or the new image's for a name the old container gives no value. An
action raises when docker refuses it or its client does not launch, naming only the docker command
and its exit, or the launch's error number, as what docker writes may carry the container's
environment; a move takes either way alike. The readbacks answer
False instead, as the runner's protocol asks. Health is read inside the container, so the adapter
holds no network address.

Design rule (K6, from the runner): every import happens when this module loads and no function
imports anything, because it runs while a release is under way.
"""

from __future__ import annotations

import json
import os
import posixpath
import subprocess
import time
from collections.abc import Callable, Mapping
from typing import Any

from hermes_cli.release_host_actions import READBACK_TIMEOUT_SECONDS
from hermes_cli.release_runner import READBACK_ATTEMPTS, READBACK_INTERVAL_SECONDS

# B3: the health answer the helper waits for, the exact body with its newline.
HEALTH_BODY = b'{"status":"ok"}\n'
# The helper names the backup after the container, with this after it.
BACKUP_SUFFIX = "-rollback"
# A move keeps the last working container, stopped, under the container's name with this after
# it, until the container that takes its place passed its health check.
KEPT_SUFFIX = "-last-working"
# A stopped container a move takes off a name and docker does not remove goes under the container's
# name with this and the short Id of that container after it, for the owner to remove.
UNREMOVED_SUFFIX = "-unremoved-"
# The helper's grace before docker stop kills the old container.
STOP_SECONDS = 30

# R4-app's read, run by node inside the owner app container: /healthz at the container's own PORT
# on its own loopback, where a redirect is no answer. It compares the status and the body's bytes
# with the helper's answer itself and tells only whether they match, by its exit, writing nothing,
# as R6's reads do: no body leaves the container. It ends itself after the readback's time limit
# too: a docker exec killed at the time limit leaves its command running in the container.
HEALTH_READ = r"""
setTimeout(() => process.exit(1), %d);
(async () => {
  const answer = await fetch(`http://127.0.0.1:${process.env.PORT}/healthz`, { redirect: "error" });
  const body = Buffer.from(await answer.arrayBuffer());
  return answer.status === 200 && body.equals(Buffer.from(%s));
})().then((ok) => process.exit(ok ? 0 : 1), () => process.exit(1));
""" % (READBACK_TIMEOUT_SECONDS * 1000, json.dumps(HEALTH_BODY.decode()))

# What docker writes, exiting nonzero, when no container or object has the name it was given.
_NO_SUCH = (b"No such container", b"No such object")
# Docker's bind propagations; a bind made without one has rprivate.
_PROPAGATIONS = ("rprivate", "private", "rshared", "shared", "rslave", "slave")
# What a bind mount's configuration may hold, all of which --mount carries; of its BindOptions,
# only the propagation.
_BIND_SETTINGS = {"Type", "Source", "Target", "ReadOnly", "BindOptions"}
_UNCARRIED_BIND = "a bind mount docker run --mount cannot make again exactly is not carried over"
# docker's env file reader holds a line of at most 64 KiB, its line feed included.
_ENV_LINE_BYTES = 64 * 1024
# The names docker's client gives a new container from its own proxy configuration, each that
# docker run does not give with --env. A name given without a value takes its value from the
# client's own environment, and stays bare, which docker reads as not set, when that has none.
_PROXIES = (
    "HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "NO_PROXY", "no_proxy",
    "FTP_PROXY", "ftp_proxy", "ALL_PROXY", "all_proxy",
)
# The seccomp options docker run takes as they are: it reads any other value as a file name.
_SECCOMP_NAMED = ("seccomp=unconfined", "seccomp:unconfined", "seccomp=builtin", "seccomp:builtin")
_UNUSABLE_BACKUP = "no backup container docker can put back as it was"
_NOT_PASSED = "the container that took the place did not pass its health check"

# What a readback meets when docker or the container does not answer as it should: no docker, a
# refusal, the time limit, or a print that is not a container record.
_UNREADABLE = (
    OSError,
    RuntimeError,
    subprocess.SubprocessError,
    ValueError,
    LookupError,
    TypeError,
    AttributeError,
)


class PutBack(RuntimeError):
    """A move whose container that takes the place did not start or did not pass its health check,
    and that then put the last working container back: it runs under the name again and passed
    its health check. The message names the refused docker command and its exit only, or the
    failed check (D-10)."""


class OwnerAppHost:
    """The owner app container's moves and readbacks on the release host, from explicit settings.

    `container` is the name the owner app container runs under and `label` the image label that
    holds its revision, both from the release settings (D-9). The backup is the container's name
    with "-rollback" after it, as the helper names it but without its time, so that a recovery in
    another process finds it; while a backup is there, docker refuses a forward's rename, before
    anything stops. Images are full image names, as the helper takes them. The other moves keep
    the last working container under the name with "-last-working" after it until the container
    that takes its place passed its health check; one left there stopped, as when docker refuses
    its removal at the end of a move, the next such move takes away first, before anything stops
    (`_park`), never removing it before the container under the name passed its health check.
    """

    def __init__(self, *, container: str, label: str) -> None:
        self.container = container
        self.label = label
        self.backup = container + BACKUP_SUFFIX
        self.kept = container + KEPT_SUFFIX

    # Moves.

    def forward(self, image: str) -> None:
        """Rename the running container to the backup name and stop it, then make the container
        anew on `image` with the running one's own settings and check its health. The settings
        and the new image (`_from_image`) are read, and refused, before anything changes. A new
        container that does not start or does not pass its health check is removed and the old
        one put back (`_take_place`); the forward itself removes no container that ran."""
        record = _inspect(self.container)
        arguments, env_file, env = _run_arguments(record, self.container, image)
        _from_image(record, _image(image), env)
        _docker("rename", self.container, self.backup)
        _docker("stop", "--time", str(STOP_SECONDS), self.backup)
        self._take_place(self.backup, True, lambda: _docker(*arguments, input=env_file, env=env))

    def back(self) -> None:
        """Put the old container back under its own name: the backup read first, then the new one
        renamed and stopped, the backup renamed back and started again, and the new one removed
        once the old one passed its health check (`_back_in_place`). Without a backup docker can
        read and put back as it was (`_usable`), back refuses before anything changes."""
        backup = _find(self.backup)
        if backup is None:
            raise RuntimeError("no backup container to go back to")
        if not _usable(backup):
            raise RuntimeError(_UNUSABLE_BACKUP)
        self._back_in_place(backup)

    def forward_repeat(self, image: str) -> None:
        """The helper's forward-repeat (D-6): the running new container renamed and stopped, made
        once more on `image` from the settings of the old container, which the backup keeps, and
        the first one removed once the second passed its health check (`_take_place`). A backup
        docker cannot put back as it was (`_usable`) refuses before anything changes, as do
        settings and a new image a forward refuses."""
        backup = _inspect(self.backup)
        if not _usable(backup):
            raise RuntimeError(_UNUSABLE_BACKUP)
        arguments, env_file, env = _run_arguments(backup, self.container, image)
        _from_image(backup, _image(image), env)

        def make() -> None:
            _docker("stop", "--time", str(STOP_SECONDS), self.kept)
            _docker(*arguments, input=env_file, env=env)

        self._park()
        self._take_place(self.kept, True, make)
        _docker("rm", "-f", self.kept)

    def remove_backup(self) -> None:
        """The helper's last step: the old container, kept as the backup, removed."""
        _docker("rm", "-f", self.backup)

    def restore(self, old_image: str) -> None:
        """The helper's recovery (B3). A container under the name on `old_image` stays. Otherwise
        the backup is read first, and only once it is there on `old_image`, in a state docker can
        put back as it was (`_usable`), is it renamed back, and started unless it runs. A
        container under the name then takes the kept name and stops first, and goes only once the
        old container passed its health check (`_back_in_place`). Without such a backup to put
        back, or when docker fails to read either name, other than to say no container has it,
        the restore refuses before anything changes. The helper's wait is the runner's readback of
        the old version."""
        current = _find(self.container)
        if current is None or current["Config"].get("Image") != old_image:
            backup = _find(self.backup)
            if backup is None or backup["Config"].get("Image") != old_image:
                raise RuntimeError("no backup container on the old image to put back")
            if not _usable(backup):
                raise RuntimeError(_UNUSABLE_BACKUP)
            if current is not None:
                self._back_in_place(backup)
                return
            _docker("rename", self.backup, self.container)
            current = backup
        if current["State"].get("Status") != "running":
            _docker("start", self.container)

    def _back_in_place(self, backup: Mapping[str, Any]) -> None:
        """The backup, read as `backup`, in place of the container under the name: that one
        renamed to the kept name and stopped, the backup renamed back and started unless it runs,
        and the one it replaced removed only once the backup passed its health check
        (`_take_place`)."""

        def start_backup() -> None:
            _docker("stop", "--time", str(STOP_SECONDS), self.kept)
            _docker("rename", self.backup, self.container)
            if backup["State"].get("Status") != "running":
                _docker("start", self.container)

        self._park()
        self._take_place(self.kept, False, start_backup)
        _docker("rm", "-f", self.kept)

    def _park(self) -> None:
        """The container under the name renamed to the kept name, before anything stops. When
        docker refuses that and a container that does not run is left under the kept name, as
        when docker refused its removal at the end of a move or a put-back failed, that one goes
        first. It is neither the container under the name nor the backup. It is removed only once
        the container under the name passed its health check, as that one then replaced it and
        passed; else, or when docker refuses its removal, it is renamed away, with
        `UNREMOVED_SUFFIX` and its short Id after the name, for the owner to remove. With no such
        container there, or a running one, the refusal is raised, before anything changes."""
        try:
            _docker("rename", self.container, self.kept)
            return
        except _Refused as refused:
            failure = refused
        left = _find(self.kept)
        if left is None or left["State"].get("Running") or left["State"].get("Status") == "running":
            raise failure
        removed = False
        if self._passes():
            try:
                _docker("rm", "-f", self.kept)
                removed = True
            except _Refused:
                pass
        if not removed:
            away = self.container + UNREMOVED_SUFFIX + str(left.get("Id", ""))[:12]
            _docker("rename", self.kept, away)
        _docker("rename", self.container, self.kept)

    def _take_place(self, kept: str, made: bool, steps: Callable[[], object]) -> None:
        """`steps` bring the container that takes the place of the last working one, kept stopped
        under `kept`, up under the name, and then its health check (`_passes`) runs. When docker
        refuses a step or the check does not pass, the last working container goes back
        (`_put_back`) and `PutBack` is raised, naming the refused docker command and its exit
        only, or the failed check. `made` tells whether the move made the container that takes the
        place, which is then removed; else it is the backup, which goes back to the backup name."""
        try:
            steps()
            failure = None if self._passes() else _NOT_PASSED
        except _Refused as refused:
            failure = str(refused)
        if failure is not None:
            self._put_back(kept, made)
            raise PutBack(failure)

    def _put_back(self, kept: str, made: bool) -> None:
        """The container under the name, if any, taken away (`_clear`), and the last working
        container given the name back from `kept`, started and checked. A refused step or a failed
        check raises here, not `PutBack`: the last working container is then not back."""
        self._clear(made)
        _docker("rename", kept, self.container)
        _docker("start", self.container)
        if not self._passes():
            raise RuntimeError("the last working container, put back, failed its health check")

    def _clear(self, made: bool) -> None:
        """The name freed of the container that took the place, if one has it: removed when the
        move made it, else stopped and renamed back to the backup name. After a refusal the name
        is read again, up to the runner's number of readback attempts, as far apart (`_wait`), so
        that this is safe to retry: no container under the name frees it; a removal docker has
        under way is waited for; and a container the move made that docker did not remove is
        stopped and renamed away, with `UNREMOVED_SUFFIX` and its short Id after the name. A
        container under the name with another Id is left alone. A name still taken then raises."""
        taking = None
        for attempt in range(READBACK_ATTEMPTS):
            if attempt:
                _wait(READBACK_INTERVAL_SECONDS)
            try:
                record = _find(self.container)
                if record is None:
                    return
                if taking is None:
                    taking = str(record.get("Id", ""))
                elif record.get("Id", "") != taking:
                    break
                if attempt and record["State"].get("Status") == "removing":
                    continue
                if made and not attempt:
                    _docker("rm", "-f", self.container)
                else:
                    _docker("stop", "--time", str(STOP_SECONDS), self.container)
                    away = self.container + UNREMOVED_SUFFIX + taking[:12] if made else self.backup
                    _docker("rename", self.container, away)
                return
            except _Refused:
                pass
        raise RuntimeError("the name is still taken, so the last working container is not back")

    def _passes(self) -> bool:
        """A move's health check of the container under the name: `health_ok`, read up to the
        runner's number of readback attempts, as far apart, until it reads true."""
        for attempt in range(READBACK_ATTEMPTS):
            if attempt:
                _wait(READBACK_INTERVAL_SECONDS)
            if self.health_ok():
                return True
        return False

    # Readbacks.

    def revision_is(self, commit: str) -> bool:
        """R1-app: the revision label of the container under the name is `commit`."""
        try:
            return self._read()["Config"]["Labels"][self.label] == commit
        except _UNREADABLE:
            return False

    def running_unrestarted(self) -> bool:
        """R3-app: the container runs and has not restarted since its start. Docker counts the
        restarts its restart policy makes from 0 at every docker start and docker run, so the
        count is unchanged since the start only while it is 0."""
        try:
            record = self._read()
            return record["State"]["Status"] == "running" and record["RestartCount"] == 0
        except _UNREADABLE:
            return False

    def health_ok(self) -> bool:
        """R4-app: Docker's health check reads "healthy", and /healthz, read inside the container,
        answers 200 with exactly the helper's body. Another status, a redirect, any other body,
        even one that differs only in white space, or no answer within the time limit reads
        False. The read compares inside the container, and only its exit comes back."""
        try:
            if self._read()["State"]["Health"]["Status"] != "healthy":
                return False
            _docker(
                "exec", self.container, "node", "-e", HEALTH_READ, timeout=READBACK_TIMEOUT_SECONDS
            )
        except _UNREADABLE:
            return False
        return True

    def _read(self) -> Any:
        return _inspect(self.container, timeout=READBACK_TIMEOUT_SECONDS)


class _Refused(RuntimeError):
    """A docker call that exited nonzero, or whose client did not launch, named by its command and
    its exit or the launch's error number only (D-10). `missing` tells whether docker said that no
    container or object has the name it was given."""

    def __init__(self, command: str, outcome: str, missing: bool = False) -> None:
        super().__init__(f"docker {command} {outcome}")
        self.missing = missing


def _docker(
    *args: str,
    timeout: float | None = None,
    input: bytes | None = None,
    env: Mapping[str, str] | None = None,
) -> bytes:
    """One docker call, as one argument list, with `input` on its standard input and `env` as the
    environment docker runs with when given, else this process's, and what it wrote to stdout. A
    nonzero exit raises naming the docker command and its exit only: the input, the environment
    and what docker writes may carry the container's environment (D-10). A launch of docker's
    client the system refuses, as fork does with EAGAIN when no process can be spared, raises the
    same way, naming the command and the error number only, and carrying no other error along."""
    try:
        done = subprocess.run(
            ["docker", *args], input=input, capture_output=True, timeout=timeout, check=False,
            env=env,
        )
    except OSError as error:
        unlaunched = error.errno
    else:
        if not done.returncode:
            return done.stdout
        missing = any(words in done.stderr for words in _NO_SUCH)
        raise _Refused(args[0], f"exited {done.returncode}", missing)
    raise _Refused(args[0], f"did not launch (errno {unlaunched})")


def _wait(seconds: float) -> None:
    """Every wait between two reads of a move's health check, or of the name a put-back frees,
    goes through here, so a test can replace it."""
    time.sleep(seconds)


def _inspect(name: str, timeout: float | None = None) -> Any:
    """The record docker inspect prints for the container `name`. Raises when there is none, and
    when docker prints anything but one container record, naming nothing of what it printed."""
    printed = _docker("inspect", "--type", "container", name, timeout=timeout)
    try:
        (record,) = json.loads(printed)
        if isinstance(record["Config"], dict) and isinstance(record["State"], dict):
            return record
    except (ValueError, TypeError, LookupError):
        pass
    raise RuntimeError("docker inspect printed no container record")


def _find(name: str) -> Any:
    """The container's record, or None when docker says no container has the name. Any other
    failure to read it raises, as the name may still hold a container."""
    try:
        return _inspect(name)
    except _Refused as refused:
        if refused.missing:
            return None
        raise


def _image(image: str) -> Any:
    """The configuration docker inspect prints for the image `image`. Raises when docker prints
    anything but one image record, naming nothing of what it printed."""
    printed = _docker("inspect", "--type", "image", image)
    try:
        (record,) = json.loads(printed)
        config = record["Config"] or {}
        if isinstance(config, dict):
            return config
    except (ValueError, TypeError, LookupError):
        pass
    raise RuntimeError("docker inspect printed no image record")


def _usable(record: Mapping[str, Any]) -> bool:
    """Whether the backup in `record` is one docker can put back as it was: exited, as a forward's
    stop leaves it, or running, as a forward whose stop failed leaves it, with docker's own flags
    agreeing: Running true in the second only, and Paused, Restarting and Dead false. A failed
    removal can leave the backup readable but dead or being removed, which docker does not start
    again. Any other status, a state docker does not give in full or that disagrees with itself,
    and one with an error docker kept from a start (State.Error, which no message names), is
    not. This is the first refusal only: a backup that still fails is put back (`_take_place`)."""
    state = record["State"]
    status = state.get("Status")
    flags = [state.get(flag) for flag in ("Running", "Paused", "Restarting", "Dead")]
    return (
        status in ("exited", "running")
        and all(isinstance(flag, bool) for flag in flags)
        and flags == [status == "running", False, False, False]
        and state.get("Error") in (None, "")
    )


def _run_arguments(
    record: Mapping[str, Any], name: str, image: str
) -> tuple[list[str], bytes | None, dict[str, str]]:
    """The docker run arguments that make the container `name` on `image` with the settings of
    the container in `record`, the env file docker run reads on its standard input and the
    environment it runs with, for the proxy settings (`_proxies`): the helper's list, each read
    as the helper reads it, and nothing more. What cannot be carried over exactly refuses before
    anything changes, naming no value: auto-remove (docker would remove the old container when it
    stops), a mount that is not a bind mount (naming only its kind), a bind --mount cannot make
    again, a seccomp option other than docker's own "unconfined" or "builtin" (docker run would
    read the profile docker inspect shows as a file name, and no weaker profile takes its place)
    and an environment entry the env file or that environment cannot hold."""
    host = record["HostConfig"]
    if host.get("AutoRemove"):
        raise RuntimeError("a container docker removes when it stops is not swapped")
    if host.get("Tmpfs"):
        raise RuntimeError("a tmpfs mount is not carried over, only bind mounts")
    for configured in host.get("Mounts") or []:
        if configured.get("Type") != "bind":
            kind = configured.get("Type")
            raise RuntimeError(f"a {kind} mount is not carried over, only bind mounts")
        options = configured.get("BindOptions") or {}
        if any(configured[key] for key in configured.keys() - _BIND_SETTINGS) or any(
            options[key] for key in options.keys() - {"Propagation"}
        ):
            raise RuntimeError(_UNCARRIED_BIND)
    arguments = ["run", "-d", "--name", name]
    if host.get("NetworkMode"):
        arguments += ["--network", host["NetworkMode"]]
    policy = host.get("RestartPolicy") or {}
    restart = policy.get("Name")
    if restart and restart != "no":
        if restart == "on-failure" and policy.get("MaximumRetryCount"):
            restart += f":{policy['MaximumRetryCount']}"
        arguments += ["--restart", restart]
    if host.get("ReadonlyRootfs"):
        arguments.append("--read-only")
    for capability in host.get("CapDrop") or []:
        arguments += ["--cap-drop", capability]
    for capability in host.get("CapAdd") or []:
        arguments += ["--cap-add", capability]
    for option in host.get("SecurityOpt") or []:
        if option.startswith("seccomp") and option not in _SECCOMP_NAMED:
            raise RuntimeError("a custom seccomp profile is not carried over")
        arguments += ["--security-opt", option]
    if host.get("PidsLimit"):
        arguments += ["--pids-limit", str(host["PidsLimit"])]
    if host.get("Memory"):
        arguments += ["--memory", str(host["Memory"])]
    if host.get("NanoCpus"):
        arguments += ["--cpus", str(host["NanoCpus"] / 1_000_000_000)]
    env, entries = _proxies(record["Config"].get("Env") or [])
    for proxy in _PROXIES:
        arguments += ["--env", proxy]
    environment = None
    if entries:
        arguments += ["--env-file", "/dev/stdin"]
        environment = _env_file(entries)
    for mount in record.get("Mounts") or []:
        if mount.get("Type") != "bind":
            raise RuntimeError(f"a {mount.get('Type')} mount is not carried over, only bind mounts")
        arguments += ["--mount", _bind(mount)]
    arguments.append(image)
    return arguments, environment, env


def _from_image(record: Mapping[str, Any], config: Mapping[str, Any], env: dict[str, str]) -> None:
    """What the new image, whose configuration is `config`, brings to the container docker run
    makes from the settings of the container in `record`. A volume the image declares at a path
    no bind mount of that container is at refuses before anything changes, as docker run would
    make an empty volume there. A proxy setting the image sets and that container gives no value
    goes into `env`, the environment docker run runs with, as docker run gives the name without a
    value and the image's value would otherwise not reach the new container; every other variable
    only the image declares docker run takes from the image itself, as with the helper today."""
    binds = {_clean(mount["Destination"]) for mount in record.get("Mounts") or []}
    if any(_clean(volume) not in binds for volume in config.get("Volumes") or {}):
        raise RuntimeError("a volume the new image declares that no bind mount covers is not made")
    image_env, _ = _proxies(config.get("Env") or [])
    for proxy in _PROXIES:
        if proxy in image_env and proxy not in env:
            env[proxy] = image_env[proxy]


def _clean(path: str) -> str:
    """`path` as docker compares a volume's path with a mount's: with no "." or ".." step and no
    doubled or trailing slash, the leading one included."""
    cleaned = posixpath.normpath(path)
    return "/" + cleaned.lstrip("/") if cleaned.startswith("/") else cleaned


def _bind(mount: Mapping[str, Any]) -> str:
    """The --mount value that makes the bind mount docker inspect shows as `mount` again: its
    source and target, read-only when it is, and its propagation when it is not docker's default,
    as one CSV record the way docker reads it, with Go's encoding/csv. A bind with more refuses:
    an SELinux label or a consistency mode, which docker inspect shows in its mode, a propagation
    docker does not name, or a line break in a path, which that reader does not give back."""
    source, target = mount["Source"], mount["Destination"]
    propagation = mount.get("Propagation") or "rprivate"
    modes = (mount.get("Mode") or "").split(",")
    if (
        any(mode not in ("", "ro", "rw", *_PROPAGATIONS) for mode in modes)
        or propagation not in _PROPAGATIONS
        or any(line_break in path for path in (source, target) for line_break in "\r\n")
    ):
        raise RuntimeError(_UNCARRIED_BIND)
    fields = ["type=bind", f"src={source}", f"dst={target}"]
    if not mount.get("RW"):
        fields.append("readonly")
    if propagation != "rprivate":
        fields.append(f"bind-propagation={propagation}")
    return ",".join(_csv_field(field) for field in fields)


def _csv_field(field: str) -> str:
    """`field` as Go's encoding/csv reads it back as one field: quoted, with each double quote
    doubled, when it holds a comma or a double quote."""
    if "," in field or '"' in field:
        return '"' + field.replace('"', '""') + '"'
    return field


def _proxies(entries: list[str]) -> tuple[dict[str, str], list[str]]:
    """The environment docker run runs with, this process's without its proxy settings and with
    those of the old container's `entries` in their place, exactly as inspected, and the entries
    that are not proxy settings, for the env file. Docker run gives each proxy name with --env,
    without a value, so that docker's client takes the old container's value from there and adds
    none of its own; a name the old container does not set, or shows bare, stays bare, which
    docker reads as not set, unless the new image sets it (`_from_image`). A proxy setting that
    environment cannot carry exactly refuses,
    naming nothing of it (D-10): a name given more than once, a value with a NUL or with what
    UTF-8 cannot encode, or an entry longer than a line of the env file."""
    env = {key: value for key, value in os.environ.items() if key not in _PROXIES}
    others, given = [], set()
    for entry in entries:
        name, equals, value = entry.partition("=")
        if name not in _PROXIES:
            others.append(entry)
            continue
        line = _utf8(entry + "\n")
        if name in given or line is None or len(line) > _ENV_LINE_BYTES or "\0" in value:
            raise RuntimeError(
                "a proxy setting docker run cannot carry exactly is not carried over"
            )
        given.add(name)
        if equals:
            env[name] = value
    return env, others


def _env_file(entries: list[str]) -> bytes:
    """The environment as the env file docker run reads: one line an entry, exactly as the
    container holds it. An entry docker's reader would not give back exactly refuses, naming
    nothing of it (D-10). The reader splits lines at a line feed and drops a carriage return
    before one, drops white space and a byte order mark before a name, skips a line that starts
    with "#", takes the value of a name with no "=" from docker's own environment, refuses an
    empty name or one with white space in it, reads only UTF-8, and holds a line of at most
    64 KiB."""
    lines = []
    for entry in entries:
        name, equals, value = entry.partition("=")
        line = _utf8(entry + "\n")
        if (
            line is None
            or len(line) > _ENV_LINE_BYTES
            or not equals
            or not name
            or name.startswith(("#", "\ufeff"))
            or any(character.isspace() for character in name)
            or "\n" in value
            or "\r" in value
        ):
            raise RuntimeError(
                "an environment entry docker's env file cannot hold is not carried over"
            )
        lines.append(line)
    return b"".join(lines)


def _utf8(text: str) -> bytes | None:
    """`text` in UTF-8, or None when it holds what UTF-8 cannot encode. The error is dropped here,
    so that no refusal carries the text along."""
    try:
        return text.encode("utf-8")
    except UnicodeEncodeError:
        return None
