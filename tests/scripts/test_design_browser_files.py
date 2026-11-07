"""Tests for the design browser files in scripts/design_browser: image, compose file, watchdog, tunnel unit, and
the runbook's sshd limits for the tunnel account with the check 3 that tries them.

Everything is read as data; the watchdog is only parsed with ``bash -n`` and, when it is installed, checked by
shellcheck. Check 3's probes are read, not run. No image is built, no container started, no tunnel opened, no
sshd changed and no network used.
"""

from __future__ import annotations

import re
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from tools.browser_supervisor_navguard import NO_NEW_WINDOWS

REPO = Path(__file__).resolve().parents[2]
DESIGN = REPO / "scripts" / "design_browser"
DOCKERFILE = DESIGN / "browser.Dockerfile"
COMPOSE = DESIGN / "compose.yaml"
WATCHDOG = DESIGN / "watchdog.sh"
TUNNEL_UNIT = DESIGN / "design-browser-tunnel.service"
README = DESIGN / "README.md"
JOURNEYS_BROWSER_DOCKERFILE = REPO / "scripts" / "journeys" / "images" / "browser.Dockerfile"

CDP_PORT = 9222
# Without it Chromium refuses Browser.getBrowserCommandLine, so the guard cannot see NO_NEW_WINDOWS.
AUTOMATION = "--enable-automation"
GUARD_COMMENT = (
    f"The navigation guard becomes active only when Browser.getBrowserCommandLine shows {NO_NEW_WINDOWS}, "
    "and with that switch no page can open a second window. This is the owner's rule for the navigation guard."
)
DIGEST_REF = re.compile(r".+@sha256:(?!(.)\1{63}$)[0-9a-f]{64}")
# The key line limits where the key may connect, not where it may listen or which Unix sockets it may reach.
TUNNEL_SSHD_MATCH = ["Match", "User", "<tunnel-user>"]
TUNNEL_SSHD_LIMITS = [["PermitListen", "none"], ["AllowStreamLocalForwarding", "no"]]
# Check 3's probes -> the exit status and a text of their refusal by OpenSSH 9.6p1 (Ubuntu 24.04).
REFUSALS = {
    "remote command": (1, ""),
    "terminal": (255, "PTY allocation request failed"),
    "forward to 127.0.0.1:22": (255, "administratively prohibited"),
    "forward to a Unix socket": (255, "connect failed"),
    "TCP listener": (255, "remote port forwarding failed for listen port"),
    "Unix-socket listener": (255, "remote port forwarding failed for listen path"),
}
MKTEMP_ASSIGNMENT = re.compile(r"(\w+)=\$\(mktemp -u (\S+)\)")


# --- readers -----------------------------------------------------------------------------------------


def base_images(path: Path) -> list[str]:
    """The image of each FROM line, flags such as --platform dropped."""
    images = []
    for line in path.read_text(encoding="utf-8").splitlines():
        words = line.split()
        if words and words[0].upper() == "FROM":
            images.append(next(word for word in words[1:] if not word.startswith("--")))
    return images


def compose() -> dict:
    return yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))


def browser_service() -> dict:
    services = compose()["services"]
    assert len(services) == 1, f"expected the design browser only, got {sorted(services)}"
    return next(iter(services.values()))


def unit_settings(path: Path) -> dict[str, dict[str, list[str]]]:
    """Section -> key -> values in file order; comments dropped, backslash continuations joined."""
    sections: dict[str, dict[str, list[str]]] = {}
    current: dict[str, list[str]] | None = None
    pending = ""
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not pending and (not line or line.startswith(("#", ";"))):
            continue
        if line.endswith("\\"):
            pending += line[:-1] + " "
            continue
        line, pending = (pending + line).strip(), ""
        header = re.fullmatch(r"\[([^\]]+)\]", line)
        if header:
            current = sections.setdefault(header.group(1), {})
            continue
        key, separator, value = line.partition("=")
        assert current is not None and separator, f"not a setting: {line!r}"
        current.setdefault(key.strip(), []).append(value.strip())
    return sections


def tunnel_command() -> list[str]:
    [command] = unit_settings(TUNNEL_UNIT)["Service"]["ExecStart"]
    return shlex.split(command)


def ssh_options(words: list[str]) -> dict[str, str]:
    """The -o Name=value options of an ssh command line, names lower-cased as ssh matches them."""
    options = {}
    for flag, option in zip(words, words[1:]):
        if flag == "-o":
            name, _, value = option.partition("=")
            options[name.lower()] = value
    return options


def code_lines_with(lines: list[str], switch: str) -> list[int]:
    """Indexes of the lines that pass the switch as a word of its own; comment lines are left out."""
    word = re.compile(rf"(?<![\w-]){re.escape(switch)}(?![\w=-])")
    return [index for index, line in enumerate(lines) if not line.lstrip().startswith("#") and word.search(line)]


def comment_above(lines: list[str], index: int) -> str:
    """The run of comment lines directly above a line, as one text with its whitespace normalised."""
    run: list[str] = []
    while index > 0 and lines[index - 1].lstrip().startswith("#"):
        index -= 1
        run.insert(0, lines[index].lstrip()[1:])
    return " ".join(" ".join(run).split())


def readme_parts() -> tuple[list[str], list[str]]:
    """The README's lines before its Checks heading, and the lines of its Checks section."""
    before, heading, rest = README.read_text(encoding="utf-8").partition("\n## Checks\n")
    assert heading, "the README has no Checks section"
    return before.splitlines(), rest.split("\n## ", 1)[0].splitlines()


def numbered_items(lines: list[str]) -> dict[int, list[str]]:
    """Number -> the lines of each item of a top-level numbered list."""
    items: dict[int, list[str]] = {}
    current: list[str] | None = None
    for line in lines:
        start = re.match(r"(\d+)\. ", line)
        if start:
            current = items.setdefault(int(start.group(1)), [])
        elif line[:1] not in {"", " "}:
            current = None
        if current is not None:
            current.append(line)
    return items


def fenced_lines(lines: list[str]) -> list[list[str]]:
    """The lines of each fenced code block, stripped, backslash continuations joined."""
    blocks: list[list[str]] = []
    inside = False
    pending = ""
    for raw in lines:
        line = raw.strip()
        if line.startswith("```"):
            inside = not inside
            if inside:
                blocks.append([])
        elif inside and line.endswith("\\"):
            pending += line[:-1] + " "
        elif inside:
            blocks[-1].append(pending + line)
            pending = ""
    return blocks


def runs(words: list[str], program: str, *arguments: str) -> bool:
    """Whether a command line runs the program with these first arguments."""
    return any(
        Path(word).name == program and words[index + 1 : index + 1 + len(arguments)] == list(arguments)
        for index, word in enumerate(words)
    )


def check_3_lines() -> list[str]:
    """The lines of check 3's code blocks, in order, backslash continuations joined."""
    _, checks = readme_parts()
    return [line for block in fenced_lines(numbered_items(checks)[3]) for line in block]


def shell_words(line: str) -> list[str]:
    """The words of a shell line, comments dropped, with ;, & and | as words of their own."""
    lexer = shlex.shlex(line, posix=True, punctuation_chars=";&|")
    lexer.whitespace_split = True
    return list(lexer)


def function_body(lines: list[str], name: str) -> list[str]:
    """The lines between `name() {` and the next line that is only `}`."""
    start = next(index for index, line in enumerate(lines) if line.startswith(f"{name}() {{"))
    return lines[start + 1 : lines.index("}", start)]


def probe_calls(lines: list[str]) -> list[tuple[int, list[str]]]:
    """The index and words of each line that calls check 3's probe helper."""
    calls = [(index, shell_words(line)) for index, line in enumerate(lines)]
    return [(index, words) for index, words in calls if words[:1] == ["probe"]]


def probe_kind(arguments: list[str]) -> str:
    """Which probe of check 3 the ssh arguments are; a listen spec that starts with $ is a variable path."""
    if "-tt" in arguments:
        return "terminal"
    for flag, spec in zip(arguments, arguments[1:]):
        if flag == "-W":
            return "forward to a Unix socket" if spec.startswith("/") else f"forward to {spec}"
        if flag == "-R":
            return "Unix-socket listener" if spec.startswith(("/", "$")) else "TCP listener"
    return "remote command" if len(arguments) > 1 and not any(word.startswith("-") for word in arguments) else "?"


def mktemp_path(lines: list[str]) -> tuple[str, str]:
    """The variable that check 3 assigns from mktemp -u, and its template."""
    assignments = [MKTEMP_ASSIGNMENT.fullmatch(line) for line in lines]
    found = [match.groups() for match in assignments if match]
    assert len(found) == 1, f"check 3 must assign one path from mktemp -u, got {found}"
    return found[0]


# --- the image ---------------------------------------------------------------------------------------


def test_the_image_is_the_journeys_browser_image_pinned_by_digest():
    bases = base_images(DOCKERFILE)
    assert bases == base_images(JOURNEYS_BROWSER_DOCKERFILE)
    assert len(bases) == 1 and DIGEST_REF.fullmatch(bases[0]), bases


# --- the watchdog ------------------------------------------------------------------------------------


def test_the_watchdog_starts_chromium_with_both_guard_switches_under_the_comment_that_says_why():
    lines = WATCHDOG.read_text(encoding="utf-8").splitlines()
    guard = code_lines_with(lines, NO_NEW_WINDOWS)
    automation = code_lines_with(lines, AUTOMATION)
    assert len(guard) == 1 and len(automation) == 1, (guard, automation)
    first, last = sorted(guard + automation)
    assert last - first <= 1, "the two switches are not next to each other"
    assert GUARD_COMMENT in comment_above(lines, first)


def test_the_watchdog_is_valid_bash():
    result = subprocess.run(["bash", "-n", str(WATCHDOG)], capture_output=True, encoding="utf-8")
    assert result.returncode == 0, result.stderr


def test_the_watchdog_passes_shellcheck():
    if shutil.which("shellcheck") is None:
        pytest.skip("shellcheck is not installed")
    result = subprocess.run(["shellcheck", str(WATCHDOG)], capture_output=True, encoding="utf-8")
    assert result.returncode == 0, result.stdout + result.stderr


# --- the compose file --------------------------------------------------------------------------------


def test_compose_publishes_the_debugging_port_on_127_0_0_1_only():
    service = browser_service()
    assert service["ports"] == [f"127.0.0.1:{CDP_PORT}:{CDP_PORT}"]
    # Host networking drops the publish and would put the forward inside the container on the host's address.
    assert "network_mode" not in service


def test_compose_drops_every_capability():
    service = browser_service()
    assert service["cap_drop"] == ["ALL"]
    assert "cap_add" not in service and not service.get("privileged")


def test_compose_forbids_new_privileges():
    options = browser_service()["security_opt"]
    assert any(re.fullmatch(r"no-new-privileges(?:[:=]true)?", option) for option in options), options


def test_compose_root_filesystem_is_read_only():
    assert browser_service()["read_only"] is True


def test_compose_mounts_no_volume():
    assert "volumes" not in browser_service()
    assert "volumes" not in compose()


def test_compose_restarts_the_browser_unless_it_was_stopped():
    assert browser_service()["restart"] == "unless-stopped"


def test_compose_gives_the_browser_1gb_of_shared_memory():
    # The 64 MB Docker default crashes the graphics process on full-page screenshots.
    assert browser_service()["shm_size"] == "1gb"


# --- the tunnel unit ---------------------------------------------------------------------------------


def test_the_tunnel_forwards_local_127_0_0_1_9222_to_the_execution_servers_127_0_0_1_9222_and_nothing_else():
    words = tunnel_command()
    assert Path(words[0]).name == "ssh"
    assert words[1:5] == ["-N", "-T", "-L", f"127.0.0.1:{CDP_PORT}:127.0.0.1:{CDP_PORT}"]
    assert [word for word in words if word in {"-L", "-R", "-D", "-W"}] == ["-L"]
    assert not {"localforward", "remoteforward", "dynamicforward"} & set(ssh_options(words))


def test_the_tunnel_exits_when_its_forward_fails_and_systemd_always_restarts_it():
    assert ssh_options(tunnel_command())["exitonforwardfailure"] == "yes"
    assert unit_settings(TUNNEL_UNIT)["Service"]["Restart"] == ["always"]


# --- the runbook -------------------------------------------------------------------------------------


def test_the_runbook_installs_sshd_limits_for_the_tunnel_account_and_check_3_tries_what_they_refuse():
    install, _ = readme_parts()
    blocks = [[line.split() for line in block] for block in fenced_lines(install)]
    match_blocks = [block for block in blocks if TUNNEL_SSHD_MATCH in block]
    assert len(match_blocks) == 1, "the install must write one sshd Match block for <tunnel-user>"
    [block] = match_blocks
    assert all(limit in block[block.index(TUNNEL_SSHD_MATCH) + 1 :] for limit in TUNNEL_SSHD_LIMITS), block

    # A broken sshd configuration must never reach a reload.
    lines = [words for block in blocks for words in block]
    validations = [index for index, words in enumerate(lines) if runs(words, "sshd", "-t")]
    reloads = [index for index, words in enumerate(lines) if runs(words, "systemctl", "reload", "ssh")]
    assert validations and reloads, "the install must run sshd -t and reload ssh"
    assert lines.index(TUNNEL_SSHD_MATCH) < validations[0] < reloads[0], "sshd -t must come before the reload"
    # The read-back, not the file, shows that the limits apply.
    readbacks = [index for index, words in enumerate(lines) if runs(words, "sshd", "-T")]
    assert readbacks, "the install must read the limits back with sshd -T"
    readback = " ".join(lines[readbacks[0]])
    assert all(name.lower() in readback for name, _ in TUNNEL_SSHD_LIMITS), readback
    assert validations[0] < readbacks[0] < reloads[0], "the read-back must come between sshd -t and the reload"

    # Check 3's Unix-socket listener uses a path from mktemp -u; read it as that template.
    variable, template = mktemp_path(check_3_lines())
    probes = [
        [word.replace(f"${variable}", template) for word in shlex.split(line, comments=True)]
        for line in check_3_lines()
    ]
    forwards = [pair for words in probes for pair in zip(words, words[1:]) if pair[0] in {"-L", "-R", "-W"}]
    reverse = [spec for flag, spec in forwards if flag == "-R"]
    assert any(not spec.startswith("/") for spec in reverse), "check 3 tries no reverse TCP listener"
    assert any(spec.startswith("/") for spec in reverse), "check 3 tries no reverse Unix-socket listener"
    assert any(flag in {"-L", "-W"} and spec.rsplit(":", 1)[-1].startswith("/") for flag, spec in forwards), (
        "check 3 tries no forward to a Unix socket"
    )


def test_check_3_judges_a_probe_only_once_ssh_shows_that_the_key_logged_in():
    body = [shell_words(line) for line in function_body(check_3_lines(), "probe")]
    [ssh] = [index for index, words in enumerate(body) if "ssh" in words]
    assert ssh_options(body[ssh])["loglevel"] == "VERBOSE", "ssh prints Authenticated to ... at VERBOSE"
    logged_in = [index for index, words in enumerate(body) if "grep" in words and "^Authenticated to " in words]
    judged = [
        index
        for index, words in enumerate(body)
        if any("$text" in word or "$want" in word or word.startswith("pass") for word in words)
    ]
    assert logged_in and judged, "the helper must look for Authenticated to and judge the text and exit status"
    assert ssh < logged_in[0] < min(judged), "the helper must find Authenticated to before it judges the probe"


def test_every_probe_of_check_3_expects_the_exit_status_and_text_of_its_refusal():
    lines = check_3_lines()
    body = set(function_body(lines, "probe"))
    assert not [line for line in lines if line not in body and "ssh" in shell_words(line)], (
        "check 3 must run ssh only through its probe helper"
    )
    probes = {}
    for _, words in probe_calls(lines):
        status, text, *arguments = words[1:]
        probes.setdefault(probe_kind(arguments), []).append((int(status), text))
    assert sorted(probes) == sorted(REFUSALS), probes
    for kind, (status, part) in REFUSALS.items():
        [(expected_status, expected_text)] = probes[kind]
        assert expected_status == status and expected_text and part in expected_text, (kind, probes[kind])
    # sshd refuses a Unix-socket forward as a failed connect, not as administratively prohibited.
    [(_, socket_text)] = probes["forward to a Unix socket"]
    assert "administratively prohibited" not in socket_text, socket_text


def test_check_3_listens_on_a_fresh_mktemp_path_and_afterwards_fails_on_and_removes_anything_there():
    lines = check_3_lines()
    variable, template = mktemp_path(lines)
    assert template.startswith("/tmp/") and template.endswith("XXX"), template
    assert [line for line in lines if line.startswith(f"{variable}=")] == [f"{variable}=$(mktemp -u {template})"], (
        f"check 3 must set {variable} only from mktemp -u"
    )
    calls = probe_calls(lines)
    [listen_spec] = [
        words[words.index("-R") + 1] for _, words in calls if probe_kind(words[3:]) == "Unix-socket listener"
    ]
    assert listen_spec.split(":", 1)[0] in {f"${variable}", f"${{{variable}}}"}, listen_spec

    last_probe = max(index for index, _ in calls)
    after = [shell_words(line) for line in lines[last_probe + 1 :]]
    assert any(runs(words, "test", "-e", f"${variable}") for words in after), "check 3 must test the path after"
    assert any(runs(words, "rm", "-f", f"${variable}") for words in after), "check 3 must remove the path after"


def test_check_3_first_connects_to_the_socket_of_its_unix_socket_forward_as_the_tunnel_account():
    lines = check_3_lines()
    calls = probe_calls(lines)
    [path] = [words[words.index("-W") + 1] for _, words in calls if probe_kind(words[3:]) == "forward to a Unix socket"]
    connects = []
    for index, line in enumerate(lines):
        words = shell_words(line)
        if runs(words, "sudo", "-u", "<tunnel-user>", "python3", "-c"):
            code = words[words.index("-c") + 1]
            if "socket.AF_UNIX" in code and f'.connect("{path}")' in code:
                connects.append(index)
    assert connects, f"check 3 must connect to {path} as <tunnel-user>"
    assert connects[0] < min(index for index, _ in calls), "the connect must come before the probes"
