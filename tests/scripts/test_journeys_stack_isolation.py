"""Tests for isolation_check.sh and isolation_check.sh server1 (card P-2).

The stand-ins and helpers are in journeys_stack_harness.
"""

from __future__ import annotations

import base64
import datetime
import http.server
import ipaddress
import json
import re
import shutil
import ssl
import threading
from pathlib import Path
from typing import NamedTuple

import pytest

from journeys_stack_harness import (
    APP_C,
    BROWSER_C,
    CHECK,
    CHECKS,
    CleanCheck,
    DASH_C,
    DECOY_CONTAINERS,
    DECOY_VOLUMES,
    F_HERMES_HOME,
    GATE_C,
    HOME_V,
    Host,
    MODEL_TOKEN,
    NO_STACK_ENV,
    PROD_C,
    PROD_ISSUER,
    PROD_ORIGIN,
    PROXY_C,
    PROXY_SOURCE,
    PROXY_V,
    SAME_HOST_REDIRECT,
    SIGN_IN_REDIRECT,
    STACK_HOST,
    all_output,
    argv_text,
    clean_check,
    cron_job,
    dispatcher_on,
    dispatcher_unset,
    docker_target,
    failed_checks,
    hermes_gives,
    host,
    parse_template,
    platform_enabled,
    profile_folder,
    reset_tree,
    resource_names_in,
    run_check,
    set_hermes_home,
    sha_tree,
    shared_reset,
    signing_key_pem,
    snapshot,
    stack,
    verdicts,
)

pytestmark = pytest.mark.linux_only


# ------------------------------------------------------------------ isolation_check.sh


def proxy_issuer() -> str:
    match = re.search(r'^ISSUER = "([^"]*)"$', PROXY_SOURCE.read_text(), re.M)
    assert match, "synthetic_proxy.py no longer has an ISSUER constant line"
    return match.group(1)


def b64d(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def add_env_line(host: Host, profile: str, line: str) -> None:
    with open(profile_folder(host, profile) / ".env", "a") as handle:
        handle.write(line + "\n")


def extra_env_name(who: str):
    return lambda host: host.update(lambda s: s["containers"][who]["env"].append("STRAY_SETTING=1"))


def extra_mount(who: str, entry: dict):
    return lambda host: host.update(lambda s: s["containers"][who]["mounts"].append(entry))


def curl_gives(**settings):
    return lambda host: host.update(lambda s: s["curl"].update(settings))


# One thing broken per case: (the check that must fail, how to break it, script environment changes).
BREAKS = {
    "extra-env-name-in-app": ("names-on-template", extra_env_name(APP_C), {}),
    "extra-env-name-in-browser": ("names-on-template", extra_env_name(BROWSER_C), {}),
    "extra-env-name-in-gateway": ("names-on-template", extra_env_name(GATE_C), {}),
    "bind-mount": ("names-on-template",
                   extra_mount(APP_C, {"type": "bind", "name": "", "dest": "/host-dir"}), {}),
    "foreign-volume": ("names-on-template",
                       extra_mount(GATE_C, {"type": "volume", "name": DECOY_VOLUMES[0], "dest": "/old"}), {}),
    "home-env-name-default": ("names-on-template", lambda h: add_env_line(h, "default", "STRAY_SETTING=1"), {}),
    "home-env-name-planner": ("names-on-template", lambda h: add_env_line(h, "planner", "STRAY_SETTING=1"), {}),
    "home-env-name-designer": ("names-on-template", lambda h: add_env_line(h, "designer", "STRAY_SETTING=1"), {}),
    "home-env-line-without-name": ("names-on-template", lambda h: add_env_line(h, "planner", "just-text"), {}),
    "home-env-line-without-name-designer": (
        "names-on-template", lambda h: add_env_line(h, "designer", "just-text"), {}),
    "dispatcher-on-default": ("dispatcher-off", dispatcher_on("default"), {}),
    "dispatcher-on-planner": ("dispatcher-off", dispatcher_on("planner"), {}),
    "dispatcher-on-designer": ("dispatcher-off", dispatcher_on("designer"), {}),
    "dispatcher-unset-planner": ("dispatcher-off", dispatcher_unset("planner"), {}),
    "dispatcher-unset-designer": ("dispatcher-off", dispatcher_unset("designer"), {}),
    "cron-job-default": ("cron-lists-empty", cron_job("default"), {}),
    "cron-job-planner": ("cron-lists-empty", cron_job("planner"), {}),
    "cron-job-designer": ("cron-lists-empty", cron_job("designer"), {}),
    "cron-command-fails": ("cron-lists-empty", hermes_gives(cron_rc=1), {}),
    "platform-in-default-config": ("no-messaging-platform", platform_enabled("default"), {}),
    "platform-in-planner-config": ("no-messaging-platform", platform_enabled("planner"), {}),
    "platform-in-designer-config": ("no-messaging-platform", platform_enabled("designer"), {}),
    "message-delivered": ("telegram-send-fails", hermes_gives(send_rc=0), {}),
    "production-answers-200": ("production-get-refused", curl_gives(code=200), {}),
    "production-answers-404": ("production-get-refused", curl_gives(code=404), {}),
    "production-answers-500": ("production-get-refused", curl_gives(code=500), {}),
    "redirect-to-the-same-host": ("production-get-refused",
                                  curl_gives(code=302, redirect=SAME_HOST_REDIRECT), {}),
    "redirect-to-the-same-host-other-case-port-user": (
        "production-get-refused",
        curl_gives(code=302, redirect="https://someone@PROD.example.invalid:8443/x"), {}),
    "redirect-to-the-same-host-trailing-dot": (
        "production-get-refused", curl_gives(code=302, redirect="https://prod.example.invalid./x"), {}),
    "redirect-without-location": ("production-get-refused", curl_gives(code=302, redirect=""), {}),
    "redirect-without-a-host": ("production-get-refused", curl_gives(code=302, redirect="/login"), {}),
    "curl-fails": ("production-get-refused", curl_gives(rc=28), {}),
    "origin-unset": ("production-get-refused", None, {"JOURNEY_PRODUCTION_ORIGIN": None}),
    "origin-empty": ("production-get-refused", None, {"JOURNEY_PRODUCTION_ORIGIN": ""}),
}


@pytest.mark.parametrize("case", sorted(BREAKS))
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


@pytest.mark.parametrize("code, redirect", [
    (401, ""), (403, ""), (302, SIGN_IN_REDIRECT), (307, SIGN_IN_REDIRECT),
    (303, "https://other.example.invalid/"),
])
def test_check_accepts_a_refusal_or_a_redirect_elsewhere(stack: Host, code: int, redirect: str):
    stack.update(lambda s: s["curl"].update(code=code, redirect=redirect))
    result = run_check(stack)
    assert result.returncode == 0, result.stdout
    assert verdicts(result)["production-get-refused"] == "PASS"


@pytest.mark.parametrize("origin", [
    "prod.example.invalid", "ftp://prod.example.invalid", "https://", 'https://prod.example.invalid/"x',
    'https://prod.example.invalid/\nheader = "x: y"', "https://prod.example.invalid/ -o /etc/x",
])
def test_check_refuses_an_origin_that_is_not_a_plain_address_without_calling_curl(stack: Host, origin: str):
    result = run_check(stack, JOURNEY_PRODUCTION_ORIGIN=origin)
    assert result.returncode == 1
    assert failed_checks(result) == {"production-get-refused"}
    assert stack.calls("curl") == []


@pytest.mark.parametrize("args", [[], ["bogus"], [""], ["SERVER2"], ["server2", "server1"]])
def test_check_prints_usage_and_exits_2_for_a_missing_or_other_argument(host: Host, args: list):
    result = host.run(CHECK, *args)
    assert result.returncode == 2
    assert "usage" in result.stderr.lower() and result.stdout == ""
    assert host.calls() == []


@pytest.mark.parametrize("stack_host", ["some-other-host.invalid", "", None])
def test_check_server2_refuses_the_wrong_host_before_any_other_check(host: Host, stack_host):
    result = host.run(CHECK, "server2", env={"JOURNEY_STACK_HOST": stack_host})
    assert result.returncode == 1
    lines = result.stdout.splitlines()
    assert len(lines) == 1 and lines[0].startswith("FAIL host")
    assert host.calls("docker") == [] and host.calls("curl") == []
    assert STACK_HOST not in all_output(result)


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


@pytest.mark.parametrize("value", ["/somewhere/else", "/opt/data/"])
@pytest.mark.parametrize("who", [DASH_C, GATE_C])
def test_check_fails_names_when_hermes_home_is_not_the_home_mount(stack: Host, who: str, value: str):
    stack.update(set_hermes_home(who, value))
    result = run_check(stack)
    assert result.returncode == 1, result.stdout
    assert list(verdicts(result)) == CHECKS
    line = next(line for line in result.stdout.splitlines() if line.startswith("FAIL names-on-template"))
    assert who in line
    if who == DASH_C:
        assert failed_checks(result) == {"names-on-template"}
    assert "names-on-template" in failed_checks(result)


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


@pytest.mark.parametrize("missing", [GATE_C, APP_C, BROWSER_C, DASH_C])
def test_check_fails_names_when_a_template_container_is_missing(stack: Host, missing: str):
    stack.update(lambda s: s["containers"].pop(missing))
    result = run_check(stack)
    assert result.returncode == 1
    assert "names-on-template" in failed_checks(result)
    assert list(verdicts(result)) == CHECKS


def test_check_does_not_pass_when_the_hermes_command_is_missing(stack: Host):
    (stack.root / "ctr/venv/hermes").unlink()
    result = run_check(stack)
    assert result.returncode == 1
    assert failed_checks(result) == {"names-on-template", "dispatcher-off", "cron-lists-empty",
                                     "no-messaging-platform", "telegram-send-fails"}


@pytest.mark.parametrize("code", [0, 2, 3, 126, 127])
def test_check_telegram_send_passes_only_when_delivery_fails(stack: Host, code: int):
    stack.update(lambda s: s["hermes"].update(send_rc=code))
    result = run_check(stack)
    assert result.returncode == 1
    assert failed_checks(result) == {"telegram-send-fails"}


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


class Seen(NamedTuple):
    method: str
    assertion: str | None
    body_bytes: int


class Refusing(http.server.BaseHTTPRequestHandler):
    """Answers every request, whatever its method, with 403 and no body, and records what arrived."""

    def __getattr__(self, name):
        if name.startswith("do_"):
            return self.refuse
        raise AttributeError(name)

    def refuse(self):
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        self.server.seen.append(Seen(self.command, self.headers.get("Cf-Access-Jwt-Assertion"), len(body)))
        self.send_response(403)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, *args):
        pass


class LocalHttps(NamedTuple):
    origin: str
    ca_bundle: Path
    seen: list[Seen]


def write_local_certificates(folder: Path) -> tuple[Path, Path, Path]:
    """A throw-away CA and a server certificate for 127.0.0.1 and localhost; returns the PEM files (ca, cert, key)."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    ca_key, server_key = ec.generate_private_key(ec.SECP256R1()), ec.generate_private_key(ec.SECP256R1())
    ca_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "journeys isolation test CA")])
    server_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")])
    now = datetime.datetime.now(datetime.timezone.utc)

    def issue(subject, key, *extensions):
        builder = (
            x509.CertificateBuilder().subject_name(subject).issuer_name(ca_name)
            .public_key(key.public_key()).serial_number(x509.random_serial_number())
            .not_valid_before(now - datetime.timedelta(hours=1)).not_valid_after(now + datetime.timedelta(days=1))
        )
        for extension, critical in extensions:
            builder = builder.add_extension(extension, critical)
        return builder.sign(ca_key, hashes.SHA256())

    ca_cert = issue(ca_name, ca_key, (x509.BasicConstraints(ca=True, path_length=None), True))
    server_cert = issue(
        server_name, server_key,
        (x509.BasicConstraints(ca=False, path_length=None), True),
        (x509.SubjectAlternativeName([x509.IPAddress(ipaddress.ip_address("127.0.0.1")), x509.DNSName("localhost")]),
         False),
    )
    folder.mkdir()
    pem = serialization.Encoding.PEM
    key_pem = server_key.private_bytes(pem, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
    files = folder / "ca.pem", folder / "server.pem", folder / "server-key.pem"
    for path, data in zip(files, (ca_cert.public_bytes(pem), server_cert.public_bytes(pem), key_pem)):
        path.write_bytes(data)
    return files


@pytest.fixture
def local_https(tmp_path: Path):
    """The endpoint in a thread of the test process, on an ephemeral port, trusted through its own CA."""
    ca, cert, key = write_local_certificates(tmp_path / "tls")
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert, key)
    server = http.server.HTTPServer(("127.0.0.1", 0), Refusing)
    server.socket = context.wrap_socket(server.socket, server_side=True)
    server.seen = []
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    try:
        yield LocalHttps(f"https://127.0.0.1:{server.server_address[1]}", ca, server.seen)
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


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
    assert stack.calls("curl") == []


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


# ------------------------------------------------------------------ isolation_check.sh server1


def test_check_server1_passes_when_production_names_another_issuer(host: Host):
    before = snapshot(host)
    result = host.run(CHECK, "server1", env=NO_STACK_ENV)
    assert (result.returncode, result.stdout, result.stderr) == (0, "PASS production-issuer-differs\n", "")
    calls = host.calls()
    assert [c["tool"] for c in calls] == ["docker"]
    assert calls[0]["argv"] == ["exec", PROD_C, "printenv", "RAPHAEL_ACCESS_ISSUER"]
    for value in (PROD_ISSUER, proxy_issuer()):
        assert value not in argv_text(host) and value not in all_output(result)
    assert snapshot(host) == before


@pytest.mark.parametrize("case", ["same", "empty", "unset", "container-missing", "container-stopped"])
def test_check_server1_fails_closed(host: Host, case: str):
    def mutate(state: dict) -> None:
        box = state["containers"][PROD_C]
        if case == "same":
            box["env"] = [f"RAPHAEL_ACCESS_ISSUER={proxy_issuer()}"]
        elif case == "empty":
            box["env"] = ["RAPHAEL_ACCESS_ISSUER="]
        elif case == "unset":
            box["env"] = ["PATH=/usr/bin"]
        elif case == "container-missing":
            state["containers"].pop(PROD_C)
        else:
            box["running"] = False
    host.update(mutate)
    result = host.run(CHECK, "server1", env=NO_STACK_ENV)
    assert result.returncode == 1
    assert re.fullmatch(r"FAIL production-issuer-differs: \S.*\n", result.stdout)
    for value in (PROD_ISSUER, proxy_issuer()):
        assert value not in all_output(result) and value not in argv_text(host)


def test_check_server1_touches_no_stack_container_and_names_only_production(host: Host):
    host.run(CHECK, "server1", env=NO_STACK_ENV)
    named = {part for c in host.calls("docker") for part in c["argv"] if part.startswith("raphael-")}
    assert named == set(parse_template()["production-container"])
