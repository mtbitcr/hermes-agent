"""Tests for isolation_check.sh (card P-2), part 3 of 3.

The stand-ins and helpers are in journeys_stack_harness and journeys_stack_isolation_support.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import pytest

from journeys_stack_harness import (
    all_output,
    argv_text,
    CHECKS,
    CleanCheck,
    DASH_C,
    DECOY_CONTAINERS,
    DECOY_VOLUMES,
    docker_target,
    F_HERMES_HOME,
    failed_checks,
    GATE_C,
    HOME_V,
    Host,
    MODEL_TOKEN,
    parse_template,
    PROD_C,
    PROD_ORIGIN,
    PROXY_C,
    PROXY_SOURCE,
    PROXY_V,
    resource_names_in,
    run_check,
    SAME_HOST_REDIRECT,
    set_hermes_home,
    sha_tree,
    SIGN_IN_REDIRECT,
    signing_key_pem,
    snapshot,
    STACK_HOST,
    verdicts,
)
from journeys_stack_isolation_support import (
    BREAKS,
    LocalHttps,
    b64d,
    proxy_issuer,
)
from journeys_stack_isolation_support import local_https  # noqa: F401
from journeys_stack_harness import clean_check, host, reset_tree, shared_reset, stack  # noqa: F401

pytestmark = pytest.mark.linux_only


@pytest.mark.parametrize("case", sorted(BREAKS)[21:])
def test_check_reports_a_broken_check_and_still_runs_the_others(stack: Host, case: str):
    broken, mutate, env = BREAKS[case]
    if mutate:
        mutate(stack)
    before = snapshot(stack)
    result = run_check(stack, **env)
    assert result.returncode == 1, result.stdout + result.stderr
    assert list(verdicts(result)) == CHECKS, "one line per check, in order"
    assert failed_checks(result) == {broken}
    assert snapshot(stack) == before
    for value in (STACK_HOST, PROD_ORIGIN, SAME_HOST_REDIRECT, SIGN_IN_REDIRECT, MODEL_TOKEN):
        assert value not in all_output(result)


def test_check_server2_passes_every_check_on_a_clean_stack(clean_check: CleanCheck):
    stack, before, result = clean_check
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.splitlines() == [f"PASS {name}" for name in CHECKS]
    assert result.stderr == ""
    assert snapshot(stack) == before


def test_check_server2_names_only_template_entries(clean_check: CleanCheck):
    stack, _, result = clean_check
    assert result.returncode == 0, result.stdout
    template = parse_template()
    allowed = set(template["containers"]) | set(template["volumes"])
    docker = stack.calls("docker")
    assert {docker_target(c)[0] for c in docker} <= {"inspect", "exec"}
    found = set().union(*(resource_names_in(part) for c in docker for part in c["argv"]))
    assert found <= allowed
    assert not found & (set(DECOY_CONTAINERS) | set(DECOY_VOLUMES) | {PROD_C})
    images = {entry["image"] for name, entry in stack.state()["containers"].items() if name in allowed}
    for call in docker:
        if "--type" in call["argv"] and call["argv"][call["argv"].index("--type") + 1] == "image":
            assert call["argv"][-1] in images, "an image id that no template container uses"


def test_check_reads_the_config_with_the_interpreter_of_the_hermes_command(clean_check: CleanCheck):
    uses = (clean_check.host.root / "interp.log").read_text().splitlines()
    assert uses.count("-") >= 5, "the served-profile, dispatcher and platform reads use that interpreter"


def test_check_reads_the_configured_hermes_home_of_the_dashboard_and_the_gateway(clean_check: CleanCheck):
    calls = clean_check.host.calls("docker")
    read = [c["argv"][-1] for c in calls if c["argv"][0] == "inspect" and F_HERMES_HOME in c["argv"]]
    assert sorted(read) == sorted([DASH_C, GATE_C])


def test_check_fails_names_when_the_gateway_uses_another_directory_with_a_stale_state(stack: Host):
    other = stack.root / "gateway-own-filesystem" / "data"
    shutil.copytree(stack.vol(HOME_V), other)
    (other / "stale-state.db").write_text("sentinel rows that must stay\n")
    stack.update(set_hermes_home(GATE_C, str(other)))
    before = sha_tree(other)
    result = run_check(stack)
    assert result.returncode == 1, result.stdout
    assert list(verdicts(result)) == CHECKS
    assert failed_checks(result) == {"names-on-template"}
    assert GATE_C in next(line for line in result.stdout.splitlines() if line.startswith("FAIL names-on-template"))
    assert sha_tree(other) == before


def test_check_fails_closed_when_the_gateway_is_not_running(stack: Host):
    stack.update(lambda s: s["containers"][GATE_C].update(running=False))
    result = run_check(stack)
    assert result.returncode == 1
    assert list(verdicts(result)) == CHECKS
    assert failed_checks(result) == set(CHECKS) - {"production-get-refused"}


def test_check_does_not_pass_when_the_hermes_command_is_missing(stack: Host):
    (stack.root / "ctr/venv/hermes").unlink()
    result = run_check(stack)
    assert result.returncode == 1
    assert failed_checks(result) == {"names-on-template", "dispatcher-off", "cron-lists-empty",
                                     "no-messaging-platform", "telegram-send-fails"}


def test_check_production_get_uses_one_assertion_that_stays_off_every_command_line(clean_check: CleanCheck):
    stack, before, result = clean_check
    assert result.returncode == 0, result.stdout + result.stderr
    minting =[c for c in stack.calls("docker") if docker_target(c) == ("exec", PROXY_C)]
    assert len(minting) == 1
    argv = minting[0]["argv"]
    assert "-i" in argv and "--user" not in argv
    after_name = argv[argv.index(PROXY_C) + 1:]
    assert after_name[:2] == ["python3", "-c"] and len(after_name) == 3
    assert "__main__" not in after_name[2]
    assert minting[0]["stdin"] == PROXY_SOURCE.read_bytes()
    jwt = minting[0]["stdout"].strip()
    assert re.fullmatch(r"[\w-]+\.[\w-]+\.[\w-]+", jwt)

    curl = stack.calls("curl")
    assert len(curl) == 1
    flags = curl[0]["argv"]
    assert flags[0] == "--disable", "curl reads a default configuration unless --disable is its first argument"
    assert flags[flags.index("-K") + 1] == "-"
    assert flags[flags.index("--max-redirs") + 1] == "0"
    assert flags[flags.index("-o") + 1] == "/dev/null"
    assert flags[flags.index("--max-time") + 1] == "20"
    assert flags[flags.index("-w") + 1] == "%{http_code} %{redirect_url}"
    assert not {"-L", "--location", "-H", "--header"} & set(flags)
    assert jwt.encode() in curl[0]["stdin"] and PROD_ORIGIN.encode() in curl[0]["stdin"]
    assert b"cf-access-jwt-assertion" in curl[0]["stdin"].lower()

    for secret in (jwt, PROD_ORIGIN, PROD_ORIGIN.split("//")[1]):
        assert secret not in argv_text(stack), "a value reached a command line"
        assert secret not in all_output(result), "a value was printed"
    assert jwt not in "".join(c.get("stderr", "") for c in stack.calls("docker"))
    assert snapshot(stack) == before, "the probe wrote to a volume"

    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding

    head, body, signature = jwt.split(".")
    key = serialization.load_pem_private_key(signing_key_pem(), password=None)
    key.public_key().verify(b64d(signature), f"{head}.{body}".encode(), padding.PKCS1v15(), hashes.SHA256())
    claims = json.loads(b64d(body))
    assert claims["iss"] == proxy_issuer() and claims["type"] == "app"


# The production probe with the real curl: the fake curl never reads a default curl configuration, so these
# tests send the probe to a throw-away HTTPS endpoint on 127.0.0.1 and hand curl a .curlrc through HOME.


@pytest.mark.parametrize("curlrc", ["", 'request = "POST"\ndata = "write=1"\n'],
                         ids=["empty-default-config", "curlrc-requests-post"])
def test_check_production_get_with_the_real_curl_is_one_get_whatever_the_default_curl_configuration(
        stack: Host, local_https: LocalHttps, tmp_path: Path, curlrc: str):
    curl = shutil.which("curl")
    assert curl, "the real curl is not on PATH"
    stack.use_real_curl(curl)
    operator_home = tmp_path / "operator-home"
    operator_home.mkdir()
    (operator_home / ".curlrc").write_text(curlrc)

    # A None value removes the variable from the environment of the check: no ambient setting decides the result.
    env = dict.fromkeys(["SSL_CERT_FILE", "SSL_CERT_DIR", "CURL_HOME", "XDG_CONFIG_HOME", "HTTPS_PROXY",
                         "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"])
    env.update(HOME=str(operator_home), CURL_CA_BUNDLE=str(local_https.ca_bundle),
               NO_PROXY="127.0.0.1,localhost", JOURNEY_PRODUCTION_ORIGIN=local_https.origin)
    result = run_check(stack, **env)

    assert result.returncode == 0, result.stdout + result.stderr
    assert verdicts(result)["production-get-refused"] == "PASS"
    minting = [c for c in stack.calls("docker") if docker_target(c) == ("exec", PROXY_C)]
    assert len(minting) == 1
    jwt = minting[0]["stdout"].strip()
    assert re.fullmatch(r"[\w-]+\.[\w-]+\.[\w-]+", jwt)

    assert [(s.method, s.body_bytes) for s in local_https.seen] == [("GET", 0)]
    assert local_https.seen[0].assertion == jwt
    assert len(stack.calls("curl")) == 1, "the wrapper did not record the one curl call"
    assert jwt not in all_output(result), "the assertion was printed"
    assert jwt not in argv_text(stack), "the assertion reached a command line"


def test_check_never_creates_a_signing_key(stack: Host):
    (stack.vol(PROXY_V) / "signing-key.pem").unlink()
    before = snapshot(stack)
    result = run_check(stack)
    assert result.returncode == 1
    assert failed_checks(result) == {"production-get-refused"}
    assert snapshot(stack) == before


@pytest.mark.parametrize("how", ["missing", "stopped"])
def test_check_production_get_fails_without_the_proxy_container(stack: Host, how: str):
    if how == "missing":
        stack.update(lambda s: s["containers"].pop(PROXY_C))
    else:
        stack.update(lambda s: s["containers"][PROXY_C].update(running=False))
    result = run_check(stack)
    assert result.returncode == 1
    assert "production-get-refused" in failed_checks(result)
    assert stack.calls("curl") == []
