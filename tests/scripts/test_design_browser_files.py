"""Tests for the design browser files in scripts/design_browser: image, compose file, watchdog, tunnel unit and
the runbook's sshd limits for the tunnel account.

Everything is read as data; the watchdog is only parsed with ``bash -n`` and, when it is installed, checked by
shellcheck. No image is built, no container started, no tunnel opened, no sshd changed and no network used.
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
    install, checks = readme_parts()
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

    check_3 = fenced_lines(numbered_items(checks)[3])
    probes = [shlex.split(line, comments=True) for block in check_3 for line in block]
    forwards = [pair for words in probes for pair in zip(words, words[1:]) if pair[0] in {"-L", "-R", "-W"}]
    reverse = [spec for flag, spec in forwards if flag == "-R"]
    assert any(not spec.startswith("/") for spec in reverse), "check 3 tries no reverse TCP listener"
    assert any(spec.startswith("/") for spec in reverse), "check 3 tries no reverse Unix-socket listener"
    assert any(flag in {"-L", "-W"} and spec.rsplit(":", 1)[-1].startswith("/") for flag, spec in forwards), (
        "check 3 tries no forward to a Unix socket"
    )
