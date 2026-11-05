"""Tests for the design browser files in scripts/design_browser: image, compose file, watchdog and tunnel unit.

Everything is read as data; the watchdog is only parsed with ``bash -n`` and, when it is installed, checked by
shellcheck. No image is built, no container started, no tunnel opened and no network used.
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
JOURNEYS_BROWSER_DOCKERFILE = REPO / "scripts" / "journeys" / "images" / "browser.Dockerfile"

CDP_PORT = 9222
# Without it Chromium refuses Browser.getBrowserCommandLine, so the guard cannot see NO_NEW_WINDOWS.
AUTOMATION = "--enable-automation"
GUARD_COMMENT = (
    f"The navigation guard becomes active only when Browser.getBrowserCommandLine shows {NO_NEW_WINDOWS}, "
    "and with that switch no page can open a second window. This is the owner's rule for the navigation guard."
)
DIGEST_REF = re.compile(r".+@sha256:(?!(.)\1{63}$)[0-9a-f]{64}")


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
