"""Tests for the journeys stack reset and isolation scripts (card P-2).

The scripts run against a fake docker, hostname, curl and sleep that sit first on PATH in a temp
folder. The "host" is a folder tree: containers and volumes are entries in a state file, volumes
are plain folders, and the shell snippets the scripts send into containers really run locally on
those folders. No docker, no real Server 2 and no outside network is involved. The fake curl never
reads a default curl configuration, so two tests of the production probe put the real curl in its
place (behind a wrapper that records its argument list) against a throw-away HTTPS endpoint on
127.0.0.1, under a temporary HOME whose .curlrc asks for a POST.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from plugins.dashboard_auth.raphael_workspace import model_policy

from journeys_stack_harness import (
    APP_C,
    APP_ENV_NAMES,
    BROWSER_C,
    CHECK,
    CHECKS,
    CleanCheck,
    DASH_C,
    DECOY_CONTAINERS,
    DECOY_VOLUMES,
    ENV_SETTINGS,
    GATE_C,
    HOME_V,
    Host,
    JOURNEYS,
    NO_STACK_ENV,
    PROD_C,
    PROXY_C,
    REAL_PYC,
    REPO,
    RESET,
    SECRET_NAMES_IN_HOME,
    STACK_CONTAINERS,
    STACK_HOME,
    STACK_VOLUMES,
    TEMPLATE,
    all_output,
    argv_text,
    clean_check,
    cron_job,
    dispatcher_on,
    dispatcher_unset,
    docker_names,
    failed_checks,
    hermes_gives,
    host,
    parse_template,
    platform_enabled,
    profile_folder,
    reset_tree,
    run_check,
    shared_reset,
    snapshot,
    stack,
    verdicts,
    write_view,
)

pytestmark = pytest.mark.linux_only


# ------------------------------------------------------------------ files in the repository


@pytest.mark.parametrize("script", [RESET, CHECK], ids=lambda p: p.name)
def test_scripts_are_executable_bash(script: Path):
    assert script.is_file(), f"{script.name} is missing"
    assert os.access(script, os.X_OK)
    assert script.read_text().splitlines()[0] == "#!/usr/bin/env bash"
    assert subprocess.run(["bash", "-n", str(script)], capture_output=True).returncode == 0


def test_template_holds_names_only():
    sections = parse_template()
    assert sections
    for section, names in sections.items():
        assert len(names) == len(set(names)), f"duplicate name in [{section}]"
        for name in names:
            assert re.fullmatch(r"[A-Za-z0-9_.-]+", name), f"[{section}] holds more than a name: {name!r}"


def test_template_lists_the_stack_names():
    sections = parse_template()
    assert sorted(sections["containers"]) == sorted(STACK_CONTAINERS)
    assert sorted(sections["volumes"]) == sorted(STACK_VOLUMES)
    assert set(sections["home-env"]) == SECRET_NAMES_IN_HOME
    assert set(sections[f"env {APP_C}"]) == set(APP_ENV_NAMES)
    assert sections[f"env {PROXY_C}"] == ["SYNTHETIC_STATE"]
    assert sections[f"env {DASH_C}"] == ["HERMES_HOME"]
    assert sections[f"env {GATE_C}"] == ["HERMES_HOME"]
    assert set(sections[f"env {BROWSER_C}"]) == {
        "JOURNEY_BASE_URL", "JOURNEY_PRODUCTION_ORIGIN", "PLAYWRIGHT_BROWSERS_PATH"}
    assert set(sections["script-input"]) == {
        "JOURNEY_STACK_HOST", "JOURNEY_PRODUCTION_ORIGIN", "JOURNEY_MODEL_TOKEN_FILE"}
    assert sections["production-container"] == [PROD_C]
    assert sections["production-read-variable"] == ["RAPHAEL_ACCESS_ISSUER"]
    expected = {"containers", "volumes", "home-env", "script-input", "production-container",
                "production-read-variable"} | {f"env {c}" for c in (APP_C, PROXY_C, DASH_C, GATE_C, BROWSER_C)}
    assert set(sections) == expected


def test_template_leaves_out_github_and_page_check_names():
    every_name = {name for names in parse_template().values() for name in names}
    assert not every_name & {"RAPHAEL_GITHUB_APP_ID", "RAPHAEL_GITHUB_INSTALLATION_ID",
                             "RAPHAEL_GITHUB_PRIVATE_KEY_FILE", "RAPHAEL_PAGE_CHECK_COMMON_NAME"}
    assert not every_name & (set(DECOY_CONTAINERS) | set(DECOY_VOLUMES))


def test_template_says_in_one_comment_line_that_every_container_needs_a_health_check():
    lines = [line for line in TEMPLATE.read_text().splitlines() if "health check" in line.lower()]
    assert len(lines) == 1 and lines[0].startswith("#"), lines
    assert "docker" in lines[0].lower() and "every container" in lines[0].lower()


def test_stack_home_sets_what_the_facts_say():
    default = yaml.safe_load((STACK_HOME / "config.yaml").read_text())
    profiles = sorted(p for p in (STACK_HOME / "profiles").iterdir() if p.is_dir())
    assert [p.name for p in profiles] == ["raphael-designer", "raphael-planner"]
    designer, planner = (yaml.safe_load((p / "config.yaml").read_text()) for p in profiles)
    multiplex = default.get("multiplex_profiles", (default.get("gateway") or {}).get("multiplex_profiles"))
    assert multiplex is True
    for config in (default, planner, designer):
        assert config["kanban"]["dispatch_in_gateway"] is False
        platforms = config.get("platforms") or {}
        assert set(platforms) <= {"api_server"}, "no messaging platform may be configured"
        assert "cron" not in config


def primary_route(profile: str):
    """The admitted base route the policy recommends for a profile, read from the policy when the test runs."""
    found = [a for (name, _provider), a in model_policy._ASSIGNMENTS.items() if name == profile and a.recommended]
    assert len(found) == 1, f"the policy has no single primary route for {profile}"
    return found[0]


def test_stack_home_enables_the_owner_workspace_and_carries_the_admitted_route(tmp_path: Path, monkeypatch):
    configs = {
        "default": STACK_HOME / "config.yaml",
        "raphael-planner": STACK_HOME / "profiles/raphael-planner/config.yaml",
    }
    def dig(node, *keys):
        for key in keys:
            node = node.get(key) if isinstance(node, dict) else None
        return node

    for profile, path in configs.items():
        config = yaml.safe_load(path.read_text())
        route = primary_route(profile)
        assert dig(config, "gateway", "api_server", "owner_workspace", "enabled") is True, profile
        assert dig(config, "model", "provider") == route.provider, profile
        assert dig(config, "model", "default") == route.model, profile
        assert dig(config, "agent", "reasoning_effort") == route.reasoning_effort, profile
        assert config.get("fallback_providers") == [], profile

    # What the plan commit reads: the policy's own reader, on a copy of the home, gives that route and no fallback.
    home = tmp_path / "home"
    shutil.copytree(STACK_HOME, home)
    monkeypatch.setenv("HERMES_HOME", str(home))
    for profile in configs:
        route = primary_route(profile)
        configured = model_policy._configured_route(profile)
        assert configured == (route.provider, route.model, route.reasoning_effort, True), profile
        provider, model, effort, fallbacks_disabled = configured
        admitted = model_policy.validate_assignment(profile, provider, model, effort, disable_fallbacks=fallbacks_disabled)
        assert admitted == route, profile


def test_stack_home_holds_no_dotenv_and_no_values():
    files = [p for p in STACK_HOME.rglob("*") if p.is_file()]
    assert files
    assert not [p for p in files if p.name == ".env"], ".gitignore ignores any file named .env"
    assert sorted(p.relative_to(STACK_HOME).as_posix() for p in files if p.name == ENV_SETTINGS) == [
        ENV_SETTINGS, f"profiles/raphael-planner/{ENV_SETTINGS}"]
    allowed = set(parse_template()["home-env"])
    for path in files:
        for line in path.read_text().splitlines():
            assigned = re.match(r"\s*([A-Z][A-Z0-9_]*)\s*=(.*)$", line)
            if assigned:
                assert assigned.group(1) in allowed, f"{path.name}: name not on the template"
                assert not assigned.group(2).strip(), f"{path.name}: a value is set"


def test_owned_paths_are_not_git_ignored(tmp_path: Path):
    assert RESET.is_file() and CHECK.is_file() and TEMPLATE.is_file(), "an owned file is missing"
    assert [p for p in STACK_HOME.rglob("*") if p.is_file()], "stack-home holds no files"
    repo = tmp_path / "ignore-check"
    repo.mkdir()
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(tmp_path),
           "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_NOSYSTEM": "1"}
    subprocess.run(["git", "init", "-q"], cwd=repo, env=env, check=True, capture_output=True)
    shutil.copy(REPO / ".gitignore", repo / ".gitignore")
    owned = [RESET, CHECK, TEMPLATE, Path(__file__).resolve()]
    owned += [p for p in STACK_HOME.rglob("*") if p.is_file()]
    for path in owned:
        rel = path.relative_to(REPO)
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text("x\n")
        result = subprocess.run(["git", "check-ignore", "-q", str(rel)], cwd=repo, env=env,
                                capture_output=True)
        assert result.returncode == 1, f"{rel} is ignored by .gitignore (or git failed)"
    # The guard itself works: a file named .env is ignored.
    (repo / "scripts/journeys/stack-home/.env").parent.mkdir(parents=True, exist_ok=True)
    (repo / "scripts/journeys/stack-home/.env").write_text("x\n")
    assert subprocess.run(["git", "check-ignore", "-q", "scripts/journeys/stack-home/.env"],
                          cwd=repo, env=env, capture_output=True).returncode == 0


# ------------------------------------------------------------------ served profiles and the real cron list

EXTRA = "extra-one"
EXTRA_KEY = "extra-profile-key-0123456789abcdef"
JOB_PROGRAM = (
    "import sys\n"
    "from cron.jobs import create_job, pause_job, update_job\n"
    "job = create_job(prompt='p', schedule='every 1h', name='journey-test')\n"
    "if sys.argv[1] == 'disabled':\n    update_job(job['id'], {'enabled': False})\n"
    "elif sys.argv[1] == 'paused':\n    pause_job(job['id'])\n"
)


def add_served_profile(host: Host, name: str = EXTRA, dispatcher: bool = True, platform: bool = False) -> Path:
    folder = host.vol(HOME_V) / "profiles" / name
    folder.mkdir(parents=True)
    (folder / ".env").write_text(f"API_SERVER_KEY={EXTRA_KEY}\n")
    config: dict = {"kanban": {"dispatch_in_gateway": dispatcher}}
    if platform:
        config["platforms"] = {"discord": {"enabled": True}}
    (folder / "config.yaml").write_text(yaml.safe_dump(config))
    return folder


def fail_line(result: subprocess.CompletedProcess, check: str) -> str:
    return next(line for line in result.stdout.splitlines() if line.startswith(f"FAIL {check}:"))


@pytest.fixture(scope="module")
def real_jobs(tmp_path_factory) -> dict[str, Path]:
    # one real cron store for each kind of job, made once with the platform's own functions
    REAL_PYC.mkdir(parents=True, exist_ok=True)
    stores = {}
    for kind in ("enabled", "disabled", "paused"):
        home = tmp_path_factory.mktemp(f"real-cron-{kind}")
        env = {"PATH": os.environ["PATH"], "HOME": str(home), "HERMES_HOME": str(home), "PYTHONPATH": str(REPO),
               "PYTHONPYCACHEPREFIX": str(REAL_PYC), "AWS_EC2_METADATA_DISABLED": "true"}
        done = subprocess.run([sys.executable, "-c", JOB_PROGRAM, kind], env=env, cwd=home,
                              capture_output=True, text=True)
        assert done.returncode == 0, done.stdout + done.stderr
        assert (home / "cron").is_dir()
        stores[kind] = home / "cron"
    return stores


@pytest.fixture
def real_stack(stack: Host) -> Host:
    hermes_gives(real_platform=True)(stack)
    return stack


def test_check_passes_the_cron_check_on_the_real_empty_job_list(real_stack: Host):
    result = run_check(real_stack)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.splitlines() == [f"PASS {name}" for name in CHECKS]


@pytest.mark.parametrize("kind, profile", [("disabled", "default"), ("disabled", "planner"),
                                           ("disabled", "designer"), ("paused", "default")])
def test_check_fails_on_any_real_cron_job_even_a_disabled_or_paused_one(real_stack: Host, real_jobs, kind, profile):
    # the reset's own `hermes project create` already made the default home's cron folder (empty); the job goes into it
    shutil.copytree(real_jobs[kind], profile_folder(real_stack, profile) / "cron", dirs_exist_ok=True)
    result = run_check(real_stack)
    name = "default" if profile == "default" else f"raphael-{profile}"
    assert result.returncode == 1, result.stdout
    assert list(verdicts(result)) == CHECKS
    assert failed_checks(result) == {"cron-lists-empty"}
    assert f"FAIL cron-lists-empty: profile {name} has a cron job" in result.stdout.splitlines()


def test_check_lists_cron_jobs_with_the_all_option(clean_check: CleanCheck):
    listed = [c for c in clean_check.host.calls("docker")
              if c["argv"][0] == "exec" and any("hermes cron list" in part for part in c["argv"])]
    assert len(listed) == 1 + len(list((clean_check.host.vol(HOME_V) / "profiles").iterdir()))
    assert all(any(part.endswith("hermes cron list --all") for part in c["argv"]) for c in listed)


def test_check_fails_each_check_of_an_extra_served_profile_by_name(stack: Host):
    add_served_profile(stack)
    cron_job(EXTRA)(stack)
    before = snapshot(stack)
    result = run_check(stack)
    assert result.returncode == 1, result.stdout
    assert list(verdicts(result)) == CHECKS
    assert failed_checks(result) == {"names-on-template", "dispatcher-off", "cron-lists-empty"}
    for check in ("names-on-template", "dispatcher-off", "cron-lists-empty"):
        assert EXTRA in fail_line(result, check)
    assert snapshot(stack) == before
    assert EXTRA_KEY not in all_output(result)


def test_check_fails_the_messaging_platform_of_an_extra_served_profile_by_name(stack: Host):
    add_served_profile(stack, dispatcher=False, platform=True)
    result = run_check(stack)
    assert result.returncode == 1, result.stdout
    assert failed_checks(result) == {"names-on-template", "no-messaging-platform"}
    line = fail_line(result, "no-messaging-platform")
    assert EXTRA in line and "discord" in line


def test_check_tries_the_send_from_an_extra_served_profile_too(stack: Host):
    add_served_profile(stack, dispatcher=False)
    hermes_gives(send_rc=0)(stack)
    result = run_check(stack)
    assert failed_checks(result) == {"names-on-template", "telegram-send-fails"}
    assert EXTRA in fail_line(result, "telegram-send-fails")


def test_check_names_an_extra_served_profile_that_has_nothing_else_wrong(stack: Host):
    add_served_profile(stack, dispatcher=False)
    result = run_check(stack)
    assert result.returncode == 1
    assert failed_checks(result) == {"names-on-template"}
    assert EXTRA in fail_line(result, "names-on-template")


@pytest.mark.parametrize("entry, kind", [
    ("Upper-Case", "dir"), ("a" * 70, "dir"), ("notes.txt", "file"), (".hidden", "dir"),
    ("bad$(touch PWNED)", "dir"),
])
def test_check_fails_closed_for_any_entry_of_the_profiles_folder_that_is_not_a_permitted_profile(stack: Host, entry, kind):
    target = stack.vol(HOME_V) / "profiles" / entry
    target.mkdir() if kind == "dir" else target.write_text("not a profile\n")
    result = run_check(stack)
    assert result.returncode == 1, result.stdout
    assert list(verdicts(result)) == CHECKS
    assert failed_checks(result) == {"names-on-template"}
    assert not list(stack.root.rglob("PWNED")), "a profile name reached a shell"
    line = fail_line(result, "names-on-template")
    if re.fullmatch(r"[A-Za-z0-9._-]+", entry):
        assert entry in line
    else:
        assert "bad;touch" not in all_output(result) and "bad$(touch" not in all_output(result)
        assert "bad;touch" not in argv_text(stack) and "bad$(touch" not in argv_text(stack)


def test_check_reports_a_served_name_that_is_not_plain_and_never_passes_it_to_a_shell(stack: Host):
    hermes_gives(serve_extra=["Bad;touch PWNED"])(stack)
    result = run_check(stack)
    assert result.returncode == 1, result.stdout
    assert list(verdicts(result)) == CHECKS
    assert failed_checks(result) == {"names-on-template"}
    assert not list(stack.root.rglob("PWNED"))
    assert "Bad;touch" not in all_output(result) and "Bad;touch" not in argv_text(stack)


def test_check_fails_names_when_the_served_profiles_cannot_be_read(stack: Host):
    hermes_gives(serve_fail=True)(stack)
    result = run_check(stack)
    assert result.returncode == 1, result.stdout
    assert list(verdicts(result)) == CHECKS
    assert failed_checks(result) == {"names-on-template"}
    assert "the served profiles cannot be read" in fail_line(result, "names-on-template")


def test_check_still_checks_both_permitted_profiles_when_the_served_profiles_cannot_be_read(stack: Host):
    hermes_gives(serve_fail=True)(stack)
    dispatcher_on("planner")(stack)
    cron_job("default")(stack)
    result = run_check(stack)
    assert failed_checks(result) == {"names-on-template", "dispatcher-off", "cron-lists-empty"}
    assert "raphael-planner" in fail_line(result, "dispatcher-off")
    assert "default" in fail_line(result, "cron-lists-empty")


def test_check_passes_the_names_check_when_only_the_permitted_profiles_exist(clean_check: CleanCheck):
    assert sorted(p.name for p in (clean_check.host.vol(HOME_V) / "profiles").iterdir()) == [
        "raphael-designer", "raphael-planner"]
    assert verdicts(clean_check.result)["names-on-template"] == "PASS"


def cron_lists(host: Host) -> list[dict]:
    return [c for c in host.calls("docker")
            if c["argv"][0] == "exec" and any("hermes cron list" in part for part in c["argv"])]


def test_check_permits_the_designer_and_checks_it_like_the_planner(clean_check: CleanCheck):
    assert (clean_check.host.vol(HOME_V) / "profiles/raphael-designer").is_dir()
    assert clean_check.result.returncode == 0, clean_check.result.stdout
    assert verdicts(clean_check.result) == {name: "PASS" for name in CHECKS}
    # One cron list for each of default, planner and designer: the designer is read like the planner.
    assert len(cron_lists(clean_check.host)) == 3


def test_check_still_passes_on_a_stack_without_the_designer(stack: Host):
    shutil.rmtree(stack.vol(HOME_V) / "profiles/raphael-designer")
    assert sorted(p.name for p in (stack.vol(HOME_V) / "profiles").iterdir()) == ["raphael-planner"]
    result = run_check(stack)
    assert result.returncode == 0, result.stdout + result.stderr
    assert verdicts(result) == {name: "PASS" for name in CHECKS}
    # Only default and planner were read: the designer is checked when the stack has it, not required.
    assert len(cron_lists(stack)) == 2


@pytest.mark.parametrize("breaking", [dispatcher_on("designer"), dispatcher_unset("designer"),
                                      cron_job("designer"), platform_enabled("designer")])
def test_check_fails_the_designer_by_name_when_it_breaks_what_the_planner_may_not(stack: Host, breaking):
    breaking(stack)
    result = run_check(stack)
    assert result.returncode == 1, result.stdout
    (failed,) = failed_checks(result)
    assert "raphael-designer" in fail_line(result, failed)


def test_check_still_checks_the_designer_when_the_served_profiles_cannot_be_read(stack: Host):
    hermes_gives(serve_fail=True)(stack)
    dispatcher_on("designer")(stack)
    result = run_check(stack)
    assert failed_checks(result) == {"names-on-template", "dispatcher-off"}
    assert "raphael-designer" in fail_line(result, "dispatcher-off")


def copy_scripts(dest: Path, rename: dict[str, str] | None = None) -> Path:
    """A copy of the scripts folder (scripts, template, stack-home, proxy), names optionally replaced."""
    target = dest / "scripts" / "journeys"
    target.mkdir(parents=True)
    for name in ("reset_stack.sh", "isolation_check.sh", "stack.env.template", "synthetic_proxy.py"):
        shutil.copy2(JOURNEYS / name, target / name)
    shutil.copytree(STACK_HOME, target / "stack-home")
    for old, new in (rename or {}).items():
        template = target / "stack.env.template"
        template.write_text(template.read_text().replace(old, new))
    return target


@pytest.mark.parametrize("edit", ["no-issuer-line", "empty-issuer"])
def test_check_server1_fails_when_the_proxy_issuer_cannot_be_read(host: Host, tmp_path: Path, edit: str):
    copy = copy_scripts(tmp_path / "copy")
    proxy = copy / "synthetic_proxy.py"
    text = proxy.read_text()
    if edit == "no-issuer-line":
        text = re.sub(r"^ISSUER = .*$", "ISSUER_RENAMED = 1", text, flags=re.M)
    else:
        text = re.sub(r'^ISSUER = ".*"$', 'ISSUER = ""', text, flags=re.M)
    proxy.write_text(text)
    result = host.run(copy / "isolation_check.sh", "server1", env=NO_STACK_ENV)
    assert result.returncode == 1
    assert result.stdout.startswith("FAIL production-issuer-differs")


# ------------------------------------------------------------------ names come only from the template


def test_both_scripts_take_every_name_from_the_template(tmp_path: Path):
    alt = Host(tmp_path / "host")
    renames = {"raphael-journeys-": "alt-rig-", PROD_C: "alt-production"}
    text = (alt.root / "state.json").read_text()
    for old, new in renames.items():
        text = text.replace(old, new)
    (alt.root / "state.json").write_text(text)
    (alt.root / "owners.json").write_text((alt.root / "owners.json").read_text().replace("raphael-journeys-", "alt-rig-"))
    write_view(json.loads(text), alt.root)
    for folder in sorted((alt.root / "volumes").iterdir()):
        if folder.name.startswith("raphael-journeys-"):
            folder.rename(folder.with_name(folder.name.replace("raphael-journeys-", "alt-rig-")))
    copy = copy_scripts(tmp_path / "copy", renames)

    reset = alt.run(copy / "reset_stack.sh", env={"JOURNEY_MODEL_TOKEN_FILE": str(alt.model_token_file())})
    assert reset.returncode == 0, reset.stdout + reset.stderr
    server2 = alt.run(copy / "isolation_check.sh", "server2")
    server1 = alt.run(copy / "isolation_check.sh", "server1")
    assert server2.returncode == 0, server2.stdout + server2.stderr
    assert server2.stdout.splitlines() == [f"PASS {name}" for name in CHECKS]
    assert server1.returncode == 0, server1.stdout + server1.stderr

    parts = [part for c in alt.calls("docker") for part in c["argv"]]
    named = set(re.findall(r"alt-rig-[a-z]+|alt-production", "\n".join(parts)))
    assert {"alt-rig-home", "alt-rig-tokens", "alt-rig-proxy", "alt-production"} <= named
    assert not set().union(*(docker_names(c) for c in alt.calls("docker"))), "an old name was used"
    assert (alt.vol("alt-rig-home") / "config.yaml").is_file()
    assert (alt.vol("alt-rig-tokens") / "workspace").is_file()
