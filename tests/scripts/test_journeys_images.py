"""Tests for the journeys stack image definitions and the owner app settings (card 1).

Everything is read as data: Dockerfile instructions, the settings lines, the template and the proxy source.
No image is built, no container started and no network used.
"""

from __future__ import annotations

import json
import posixpath
import re
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
JOURNEYS = REPO / "scripts" / "journeys"
IMAGES = JOURNEYS / "images"
PROXY_DOCKERFILE = IMAGES / "proxy.Dockerfile"
BROWSER_DOCKERFILE = IMAGES / "browser.Dockerfile"
APP_CA_DOCKERFILE = IMAGES / "app-ca.Dockerfile"
DOCKERFILES = [PROXY_DOCKERFILE, BROWSER_DOCKERFILE, APP_CA_DOCKERFILE]
SETTINGS = JOURNEYS / "stack-app.settings"
TEMPLATE = JOURNEYS / "stack.env.template"
PROXY_SOURCE = JOURNEYS / "synthetic_proxy.py"
S6_DASHBOARD_RUN = REPO / "docker" / "s6-rc.d" / "dashboard" / "run"
OWNER_WRITE_WATCH = REPO / "hermes_cli" / "owner_write_watch.py"

BOUND_PROJECT = "journey-test-project"
BOUND_BOARD = "journey-test"
PLAYWRIGHT_VERSION = "1.63.0"

SLUG = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
PLAIN_FILE_NAME = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*")
DIGEST_REF = re.compile(r".+@sha256:(?!(.)\1{63}$)[0-9a-f]{64}")


# --- readers -----------------------------------------------------------------------------------------


def parse_template() -> dict[str, list[str]]:
    """Sections of the template, with the rules reset_stack.sh applies in read_template."""
    sections: dict[str, list[str]] = {}
    current = None
    for raw in TEMPLATE.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        header = re.fullmatch(r"\[([^\]]+)\]", line)
        if header:
            current = header.group(1)
            sections.setdefault(current, [])
        else:
            assert current is not None, f"name outside a section: {line!r}"
            assert re.fullmatch(r"[A-Za-z0-9_.-]+", line), f"[{current}] holds more than a name: {line!r}"
            sections[current].append(line)
    return sections


def template_container(role: int) -> str:
    """Roles are the order of [containers]: proxy, owner app, dashboard, gateway, browser."""
    return parse_template()["containers"][role]


def read_settings() -> dict[str, str]:
    """NAME=value lines; blank lines and # comments are skipped; the value is everything after the first =."""
    settings: dict[str, str] = {}
    for number, raw in enumerate(SETTINGS.read_text().splitlines(), 1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        name, separator, value = raw.partition("=")
        assert separator and re.fullmatch(r"[A-Z][A-Z0-9_]*", name), f"line {number} is not NAME=value"
        assert name not in settings, f"{name} is set twice"
        settings[name] = value
    return settings


def load_proxy_module() -> types.ModuleType:
    """Run the proxy source as a module without importing it (no bytecode file, nothing is served)."""
    module = types.ModuleType("synthetic_proxy_read_for_images_test")
    module.__file__ = str(PROXY_SOURCE)
    exec(compile(PROXY_SOURCE.read_text(), str(PROXY_SOURCE), "exec"), module.__dict__)
    return module


def proxy_ca_file_name() -> str:
    """The certificate file the proxy keeps in its state folder (the public one, not the key)."""
    names = re.findall(r'state / "([^"]+)"', PROXY_SOURCE.read_text())
    public = {name for name in names if name.startswith("ca") and "key" not in name}
    assert len(public) == 1, f"expected one public CA file name in the proxy source, got {public}"
    return public.pop()


def proxy_state_folder() -> str:
    found = re.search(r'os\.environ\.get\("SYNTHETIC_STATE", "([^"]+)"\)', PROXY_SOURCE.read_text())
    assert found, "the proxy source has no default state folder"
    return found.group(1)


def parse_dockerfile(path: Path) -> list[tuple[str, str]]:
    """Instructions as (NAME, arguments); comments dropped, backslash continuations joined."""
    logical: list[str] = []
    pending = ""
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if line.startswith("#") or (not line and not pending):
            continue
        if line.endswith("\\"):
            pending += line[:-1] + " "
            continue
        logical.append((pending + line).strip())
        pending = ""
    if pending.strip():
        logical.append(pending.strip())
    return [(item.split(None, 1)[0].upper(), item.split(None, 1)[1] if " " in item else "") for item in logical]


def of_kind(path: Path, kind: str) -> list[str]:
    return [args for name, args in parse_dockerfile(path) if name == kind]


def copy_parts(args: str) -> tuple[list[str], str]:
    """(sources, destination) of a COPY or ADD line, flags dropped."""
    words = [word for word in args.split() if not word.startswith("--")]
    return words[:-1], words[-1]


def global_args(path: Path) -> dict[str, str | None]:
    """ARG names with their defaults, for the ARG lines before the first FROM."""
    found: dict[str, str | None] = {}
    for name, args in parse_dockerfile(path):
        if name == "FROM":
            break
        if name == "ARG":
            key, separator, value = args.partition("=")
            found[key.strip()] = value.strip() if separator else None
    return found


def from_images(path: Path) -> list[str]:
    images = []
    for args in of_kind(path, "FROM"):
        words = [word for word in args.split() if not word.startswith("--")]
        images.append(words[0])
    return images


def looks_secret(text: str) -> bool:
    if re.search(r"[A-Za-z0-9]{32,}", text):
        return True
    for run in re.findall(r"[A-Za-z0-9+/_=-]{32,}", text):
        if re.search(r"[0-9]", run) and re.search(r"[a-z]", run) and re.search(r"[A-Z]", run):
            return True
    return bool(re.search(r"sk-|ghp_|github_pat_|eyJ|BEGIN [A-Z ]*KEY", text))


# --- fact 4: the owner app's membership and binding rules ---------------------------------------------


def is_placeholder(text: str) -> bool:
    lowered = text.strip().lower()
    return (
        lowered in {"placeholder", "changeme", "change-me", "replace-me", "example"}
        or lowered.startswith("your-")
        or any(mark in lowered for mark in ("<", ">", "${"))
    )


def good_text(value) -> bool:
    return isinstance(value, str) and bool(value.strip()) and not is_placeholder(value)


def good_slug(value) -> bool:
    return isinstance(value, str) and len(value) <= 64 and SLUG.fullmatch(value) is not None


def check_bindings(raw: str) -> set[str]:
    bindings = json.loads(raw)
    assert isinstance(bindings, list) and bindings, "bindings must be a non-empty array"
    project_ids: list[str] = []
    route_slugs: list[str] = []
    for entry in bindings:
        assert good_text(entry["projectId"]), "projectId is blank or a placeholder"
        project_ids.append(entry["projectId"])
        assert good_slug(entry["routeSlug"]), "routeSlug is not a slug"
        route_slugs.append(entry["routeSlug"])
        hermes = entry["sources"]["hermes"]
        assert good_slug(hermes["projectSlug"]) and good_slug(hermes["boardSlug"])
        documents = entry["sources"]["documents"]
        assert isinstance(documents, list) and len(documents) <= 1
        for document in documents:
            assert document["kind"] == "github"
            assert good_text(document["owner"]) and good_text(document["repo"])
    assert len(set(project_ids)) == len(project_ids), "projectId is not unique"
    assert len(set(route_slugs)) == len(route_slugs), "routeSlug is not unique"
    return set(project_ids)


def check_membership(raw: str, issuer: str, bound_projects: set[str]) -> list[dict]:
    memberships = json.loads(raw)
    assert isinstance(memberships, list) and memberships, "membership must be a non-empty array"
    ids = []
    for entry in memberships:
        assert good_text(entry["membershipId"]), "membershipId is blank or a placeholder"
        ids.append(entry["membershipId"])
        assert entry["issuer"] == issuer, "issuer differs from RAPHAEL_ACCESS_ISSUER"
        assert good_text(entry["subject"]), "subject is blank or a placeholder"
        assert isinstance(entry["status"], str) and entry["status"].strip()
        assert entry["role"] == "owner"
        project_ids = entry["projectIds"]
        assert isinstance(project_ids, list) and project_ids
        assert all(good_text(item) for item in project_ids)
        assert len(set(project_ids)) == len(project_ids)
    assert len(set(ids)) == len(ids), "membershipId is not unique"
    active = [entry for entry in memberships if entry["status"] == "active"]
    assert len(active) == 1, "exactly one membership must be active"
    assert set(active[0]["projectIds"]) <= bound_projects, "an active project id has no binding"
    return active


# --- the six tests --------------------------------------------------------------------------------------


def test_every_settings_name_is_in_the_owner_app_section_of_the_template():
    app_section = f"env {template_container(1)}"
    allowed = parse_template()[app_section]
    names = list(read_settings())
    assert names, "the settings file holds no setting"
    assert [name for name in names if name not in allowed] == []


def test_issuer_audience_membership_and_addresses_match_the_stack_files():
    settings = read_settings()
    proxy = load_proxy_module()
    assert settings["RAPHAEL_ACCESS_ISSUER"] == proxy.ISSUER
    assert settings["RAPHAEL_ACCESS_AUDIENCE"] == proxy.AUDIENCE
    assert re.fullmatch(r"https://[a-z0-9-]+\.cloudflareaccess\.com/?", settings["RAPHAEL_ACCESS_ISSUER"])
    assert settings["RAPHAEL_PUBLIC_ORIGIN"] == f"https://{proxy.WORKSPACE_HOST}"

    bound = check_bindings(settings["RAPHAEL_PROJECT_BINDINGS_JSON"])
    assert BOUND_PROJECT in bound
    binding = next(
        entry for entry in json.loads(settings["RAPHAEL_PROJECT_BINDINGS_JSON"])
        if entry["projectId"] == BOUND_PROJECT
    )
    assert binding["sources"]["hermes"]["boardSlug"] == BOUND_BOARD
    active = check_membership(settings["RAPHAEL_OWNER_MEMBERSHIP_JSON"], settings["RAPHAEL_ACCESS_ISSUER"], bound)
    assert active[0]["subject"] == proxy.SUBJECT
    assert BOUND_PROJECT in active[0]["projectIds"]

    dashboard_port = re.search(r"HERMES_DASHBOARD_PORT:-(\d+)", S6_DASHBOARD_RUN.read_text()).group(1)
    api_port = re.search(r"^DEFAULT_PORT = (\d+)$", OWNER_WRITE_WATCH.read_text(), re.M).group(1)
    api_host = re.search(r'^DEFAULT_HOST = "([^"]+)"$', OWNER_WRITE_WATCH.read_text(), re.M).group(1)
    assert api_host == "127.0.0.1"
    assert settings["HERMES_API_BASE_URL"] == f"http://{template_container(2)}:{dashboard_port}"
    assert settings["HERMES_CONVERSATION_API_BASE_URL"] == f"http://{api_host}:{api_port}"


def test_the_seven_token_file_values_are_distinct_plain_names_under_one_folder():
    settings = read_settings()
    token_names = [name for name in parse_template()[f"env {template_container(1)}"] if name.endswith("_TOKEN_FILE")]
    assert len(token_names) == 7
    assert [name for name in token_names if name not in settings] == []
    values = [settings[name] for name in token_names]
    assert len(set(values)) == 7, "two variables share a token file"
    folders = {posixpath.dirname(value) for value in values}
    assert len(folders) == 1, f"token files are not under one folder: {folders}"
    folder = folders.pop()
    assert folder.startswith("/") and folder != "/" and ".." not in folder.split("/")
    for value in values:
        assert PLAIN_FILE_NAME.fullmatch(posixpath.basename(value)), value


def test_no_value_looks_like_a_key_or_token():
    settings = read_settings()
    for name, value in settings.items():
        assert not (name.endswith("TOKEN") or name.endswith("KEY")) or not value, f"{name} carries a value"
        assert not looks_secret(value), f"{name} has a secret-looking value"
    for path in DOCKERFILES:
        text = re.sub(r"sha256:[0-9a-f]{64}", "", path.read_text())
        assert not looks_secret(text), f"{path.name} holds a secret-looking word"


def test_each_dockerfile_base_is_pinned_by_digest():
    for path in DOCKERFILES:
        bases = from_images(path)
        assert bases, f"{path.name} has no FROM"
        defaults = global_args(path)
        for base in bases:
            variable = re.fullmatch(r"\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?", base)
            if variable is None:
                assert DIGEST_REF.fullmatch(base), f"{path.name}: {base} is not pinned by digest"
                continue
            name = variable.group(1)
            assert name in defaults, f"{path.name}: {base} is not a global ARG"
            if defaults[name] is not None:
                assert DIGEST_REF.fullmatch(defaults[name]), f"{path.name}: {name} default is not a digest"
            else:
                assert name.endswith("_DIGEST"), f"{path.name}: {name} has no default and is not a digest argument"


def test_browser_installs_playwright_exactly_and_the_images_agree_on_the_proxy_ca():
    ca_name = proxy_ca_file_name()
    settings = read_settings()

    # Browser: Playwright pinned exactly, the page-check not copied in.
    assert from_images(BROWSER_DOCKERFILE) and f"playwright:v{PLAYWRIGHT_VERSION}" in from_images(BROWSER_DOCKERFILE)[0]
    runs = of_kind(BROWSER_DOCKERFILE, "RUN")
    pinned = [version for run in runs for version in re.findall(r"playwright@(\S+)", run)]
    assert pinned and set(pinned) == {PLAYWRIGHT_VERSION}
    assert any(re.search(r"npm\s+(install|ci)\b", run) for run in runs)
    for kind in ("COPY", "ADD"):
        for args in of_kind(BROWSER_DOCKERFILE, kind):
            assert "page-check" not in args

    # Browser: the CA reaches the system store and the NSS database of the image's default user.
    assert of_kind(BROWSER_DOCKERFILE, "USER") == []
    sources, store_file = copy_parts(of_kind(BROWSER_DOCKERFILE, "COPY")[0])
    assert sources == [ca_name]
    assert posixpath.dirname(store_file) == "/usr/local/share/ca-certificates" and store_file.endswith(".crt")
    assert any("update-ca-certificates" in run for run in runs)
    assert any("libnss3-tools" in run for run in runs)
    commands = [part.strip() for run in runs for part in run.split("&&")]
    imports = [part for part in commands if re.match(r"certutil\b", part) and re.search(r"\s-A\b", part)]
    assert len(imports) == 1
    assert re.search(r'-d\s+"?sql:(\$HOME|\$\{HOME\}|~)/\.pki/nssdb"?', imports[0])
    assert re.search(r'-t\s+"?C,,', imports[0])
    assert re.search(rf"-i\s+{re.escape(store_file)}(\s|$)", imports[0])

    # Owner app layer: only that public file, nothing else added, and NODE_EXTRA_CA_CERTS points at it.
    app_ca = parse_dockerfile(APP_CA_DOCKERFILE)
    assert {name for name, _args in app_ca} == {"ARG", "FROM", "COPY"}
    app_sources, app_file = copy_parts(of_kind(APP_CA_DOCKERFILE, "COPY")[0])
    assert app_sources == [ca_name]
    assert len(of_kind(APP_CA_DOCKERFILE, "COPY")) == 1
    assert settings["NODE_EXTRA_CA_CERTS"] == app_file

    # Proxy: a non-root user owns the folder where the proxy writes ca.pem; the proxy source is the only file in.
    proxy_dockerfile = parse_dockerfile(PROXY_DOCKERFILE)
    users = [args for name, args in proxy_dockerfile if name == "USER"]
    assert users and users[-1].split(":")[0] not in {"root", "0"}
    user = users[-1].split(":")[0]
    state = proxy_state_folder()
    runs = of_kind(PROXY_DOCKERFILE, "RUN")
    assert any(state in run and re.search(rf"chown\b.*\b{re.escape(user)}\b", run) for run in runs)
    copies = [copy_parts(args) for args in of_kind(PROXY_DOCKERFILE, "COPY")]
    assert [sources for sources, _dest in copies] == [[PROXY_SOURCE.name]]
    assert PROXY_SOURCE.is_file()
    cryptography = [run for run in runs if "cryptography" in run]
    assert cryptography and all(re.search(r"cryptography==\d+(\.\d+)*(\s|$)", run) for run in cryptography)
    assert [json.loads(args) for args in of_kind(PROXY_DOCKERFILE, "CMD")] == [["python3", PROXY_SOURCE.name]]

    # The private key never enters any image: no wildcard or whole-folder source, no key file name.
    for path in DOCKERFILES:
        for kind in ("COPY", "ADD"):
            for args in of_kind(path, kind):
                for source in copy_parts(args)[0]:
                    assert source not in {".", "./", "/"} and "*" not in source and "key" not in source.lower()


def test_a_digest_must_be_exactly_64_lowercase_hex_characters():
    good = "a1b2c3d4e5f60718293a4b5c6d7e8f90a1b2c3d4e5f60718293a4b5c6d7e8f90"
    assert DIGEST_REF.fullmatch(f"image:1@sha256:{good}")
    for bad in (good[:-1], good + "0", good.upper(), "g" + good[1:], ""):
        assert DIGEST_REF.fullmatch(f"image:1@sha256:{bad}") is None, bad


def test_a_digest_made_of_one_repeated_character_is_refused():
    for char in "0af":
        assert DIGEST_REF.fullmatch(f"image:1@sha256:{char * 64}") is None, char
